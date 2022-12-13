import re
import json
import itertools

from typing import Callable

KEYWORDS = [
    "Transcriber's Note",
    "CONTENTS",
    "Copyright",
    "copyright",
    "ISBN",
    "EBOOK",
    "©",
]


def rstrip(line):
    return re.sub(r"^ +", "", line, flags=re.MULTILINE)


def filter_pargraphs(paragraph):
    return (
        len(re.findall(r"[a-zA-Z]", paragraph)) > (len(paragraph) / 3)
        and not any(keyword in paragraph for keyword in KEYWORDS)
        and len(paragraph) > 5
    )


def process_jsonl(jsonline, processor: Callable) -> str:
    """
    Processes a json encoded string and returns it as json string.
    """
    raw_book = json.loads(jsonline)
    return json.dumps(processor(raw_book), ensure_ascii=False)


naive_segmenter = re.compile(r"(\.\"|\.\)|\.|\?|\!)(?![\)\'\"\.])")


def naive_sentence_split(line, max_len):
    current_position, segments = 0, []
    sentence_iterator = naive_segmenter.finditer(line)

    while current_position < len(line):
        *_, end_position = itertools.chain(
            [None,],
            itertools.takewhile(
                lambda x: x.end() - current_position < max_len, sentence_iterator
            ),
        )
        segments.append(
            line[
                current_position : end_position.end() if end_position else max_len
            ].strip()
        )
        current_position = (
            end_position.end() if end_position else current_position + max_len
        )

    return segments
