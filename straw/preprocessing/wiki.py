from .webcrawl import process_webcrawls
from typing import List


def process_wiki(raw_text) -> List[str]:
    paragraphs = process_webcrawls(raw_text)
    for w in ["See also", "References", "External links"]:
        paragraphs = [p for p in paragraphs if not p.startswith(w)]

    return paragraphs
