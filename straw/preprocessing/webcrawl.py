import re
from typing import List

from .utils import filter_pargraphs, rstrip

reg = re.compile(
    r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def process_webcrawls(raw_text) -> List[str]:
    """
    Processes a webcrawled text and returns list of cleaned paragraphs.
    """
    # italics
    text = rstrip(re.sub(r"_(.*?)_", r" \1 ", raw_text))
    text = rstrip(re.sub(r"\[(.*?)\]", r" \1 ", text))

    # Remove >
    text = rstrip(re.sub(r"^\>", " ", text, flags=re.MULTILINE))

    # Remove ** Bold ** markers
    text = rstrip(re.sub(r"(?:\*\*)", " ", text))

    # Clean
    text = rstrip(reg.sub(" ", text))

    # Split into chapters and paragraphs
    chapters = re.compile(r"\n\n+", re.MULTILINE).split(text)
    chapters = [
        re.compile("\n", re.MULTILINE).split(chapter)
        for chapter in chapters
        if len(chapter) > 5
    ]

    # Remove redundant chapters (ratio of non alphabetics,
    # or shorter chapters maybe, contains any of the keywords)
    chapters = [list(filter(filter_pargraphs, chapter)) for chapter in chapters]

    # Join paragraphs
    chapters = [rstrip("\n".join(chapter)).strip() for chapter in chapters]

    # Clean double periods
    chapters = [rstrip(re.sub(r"\ +", " ", chapter)) for chapter in chapters]

    return chapters
