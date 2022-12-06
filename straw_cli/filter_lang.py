import json
import itertools
import argparse
import multiprocessing as mp

from tqdm import tqdm
from straw.filtering import LanguageFilter

language_filter = LanguageFilter()


def predict_jsonstrs(lines):
    # Take only the first 2**19 characters,
    # to avoid OOM errors and this is informative
    # enough for the language filter
    docs = []
    for l in lines:
        if l is not None:
            docs.append(json.loads(l)[: 2 ** 19])

    # Predict the language of the documents
    predictions = language_filter(docs)
    lines = lines[: len(docs)]
    return [l for l, p in zip(lines, predictions) if p]


def cli_main():

    argparser = argparse.ArgumentParser(
        "straw-filter-lang",
        description="Filter out non-english docs in a jsonl file containing docs, one json encoded doc per line.",
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
        "--chunksize", type=int, default=8, help="Number of docs to process per worker",
    )
    argparser.add_argument(
        "--total-docs",
        type=int,
        default=None,
        help="Number of docs to process (for logging progress)",
    )

    args = argparser.parse_args()

    indir = args.input_jsonl
    outdir = args.output_jsonl
    nworkers = args.nworkers
    chunk_size = args.chunksize
    total_docs = args.total_docs // chunk_size if args.total_docs is not None else None

    with open(outdir, "w") as fo, open(indir) as fi:
        with mp.Pool(processes=nworkers) as pool:
            # We use imap instead of map because imap is lazy
            # and can keep the order of the input.
            processors_iter = pool.imap(
                predict_jsonstrs,
                tqdm(
                    itertools.zip_longest(*[fi] * chunk_size),
                    desc="Filtering docs",
                    total=total_docs,
                ),
            )
            for books in processors_iter:
                fo.write("".join(books))
