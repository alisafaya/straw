import json
import argparse
import contextlib
import itertools

import multiprocessing as mp

from tqdm import tqdm

from straw.normalizer import TextNormalizer
from straw.filtering import LanguageFilter
from straw.preprocessing import (
    process_gutenberg,
    process_books3,
    process_webcrawls,
    process_wiki,
    naive_sentence_split,
)


class StrawProcessor(object):
    def __init__(self, args):
        self.args = args
        self.subsets = args.pile_subsets.split(",")

        self.preprocessors = {
            "Books3": process_books3,
            "Gutenberg (PG-19)": process_gutenberg,
            "OpenWebText2": process_webcrawls,
            "Pile-CC": process_webcrawls,
            "Wikipedia (en)": process_wiki,
        }

        for subset in self.subsets:
            if subset not in self.preprocessors:
                raise ValueError("Subset {} not supported".format(subset))

        self.chunk_length = self.args.text_chunk_size
        self.min_len = self.args.min_text_length
        self.max_unk_ratio = self.args.max_unk_ratio
        self.chunk_text = self.args.chunk_text

    def initialize(self):
        global normalizer
        global language_filter
        normalizer = TextNormalizer()
        language_filter = LanguageFilter()

    def process(self, lines):

        results = []
        for line in lines:
            try:
                if line is None:
                    continue
                json_obj = json.loads(line)
                subset_name = json_obj["meta"]["pile_set_name"]
                text = json_obj["text"]
            except json.JSONDecodeError as e:
                print(
                    "Couldn't parse input, please make sure a valid jsonl file from Pile is present."
                )
                raise e

            if subset_name not in self.subsets or len(text) < self.min_len:
                continue

            # Apply language filter
            unk_ratio = language_filter.get_unk_ratios([text])
            if unk_ratio > self.max_unk_ratio:
                continue

            # Apply preprocessing according to subset
            paragraphs = self.preprocessors[subset_name](text)

            # Apply normalizer
            paragraphs = normalizer(paragraphs)

            # Apply sentence splitter
            if self.chunk_text:
                nparagraphs = []
                for paragraph in paragraphs:
                    nparagraphs.extend(
                        naive_sentence_split(paragraph, self.chunk_length)
                    )
                paragraphs = nparagraphs

            paragraphs = [p for p in paragraphs if len(p) > 5]

            # Filter out too short samples
            total_len = sum([len(p) for p in paragraphs])
            if total_len < self.min_len:
                continue

            results.append(
                json.dumps(
                    {"subset": subset_name, "text": "\n".join(paragraphs)}, ensure_ascii=False
                )
            )

        if len(results) > 0:
            return "\n".join(results)

        return ""


def cli_main():

    argparser = argparse.ArgumentParser(
        "straw-process-pile", description="Process the Pile jsonl files.",
    )

    argparser.add_argument(
        "--input-jsonl", type=str, help="Path of the jsonl file containing docs.",
    )
    argparser.add_argument(
        "--output-jsonl",
        type=str,
        help="Directory to save the output in the same format as the input.",
    )
    argparser.add_argument(
        "--nworkers", type=int, default=8, help="Number of workers",
    )
    argparser.add_argument(
        "--read-chunk-size",
        type=int,
        default=8,
        help="Number of docs to process per worker",
    )
    argparser.add_argument(
        "--total-docs",
        type=int,
        default=None,
        help="Number of docs to process (for logging progress)",
    )
    argparser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Minimum length of text to keep (in characters)",
    )
    argparser.add_argument(
        "--chunk-text",
        type=bool,
        default=False,
        help="Whether to chunk the text into sentences or paragraphs",
    )
    argparser.add_argument(
        "--text-chunk-size",
        type=int,
        default=1024,
        help="Number of characters to chunk the text into",
    )
    argparser.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.05,
        help="Maximum ratio of unknown tokens to keep a sample (English only)",
    )
    argparser.add_argument(
        "--pile-subsets",
        type=str,
        default="Books3,Gutenberg (PG-19),OpenWebText2,Pile-CC,Wikipedia (en)",
        help="Comma separated list of Pile subsets to keep and process",
    )

    args = argparser.parse_args()
    total_docs = (
        args.total_docs // args.read_chunk_size if args.total_docs is not None else None
    )


    with contextlib.ExitStack() as stack:
        input = stack.enter_context(open(args.input_jsonl, "r", encoding="utf-8"))
        output = stack.enter_context(open(args.output_jsonl, "w", encoding="utf-8"))

        straw_processor = StrawProcessor(args)
        pool = mp.Pool(args.nworkers, initializer=straw_processor.initialize)

        processed_docs = pool.imap_unordered(
            straw_processor.process,
            itertools.zip_longest(*[input] * args.read_chunk_size),
            chunksize=8
        )

        for doc_jsons in tqdm(
            processed_docs, desc="Processing docs", total=total_docs,
        ):
            if doc_jsons:
                print(doc_jsons, file=output, flush=True)
