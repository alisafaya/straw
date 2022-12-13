import re
from typing import List
from .utils import filter_pargraphs, rstrip


def process_books3(raw_book) -> List[str]:
    """
    Processes a books3 file and returns cleaned raw text.
    """

    # Remove lists
    book = rstrip(
        re.sub(
            r"(^(?:\*\*).{1,100}(?:\*\*).{0,100}\n+){2,}",
            "",
            raw_book,
            flags=re.MULTILINE,
        )
    )
    book = rstrip(re.sub(r"(^\d.{0,100}\n+){3,}", "", book, flags=re.MULTILINE))
    book = rstrip(re.sub(r"(^.{0,20}\n+){10,}", "\n", book, flags=re.MULTILINE))

    # Clean encoding errors
    # Replace invalid quotes with valid ones
    book = rstrip(re.sub(r"â[¦]", "'", book))
    book = rstrip(re.sub(r"Ã©", "é", book))

    # italics
    book = rstrip(re.sub(r"_(.*?)_", r" \1 ", book))
    book = rstrip(re.sub(r"\[(.*?)\]", r" \1 ", book))

    # Remove illustrations
    book = rstrip(re.sub(r"^\[.*(\n.*){0,10}\]", " ", book, flags=re.MULTILINE))

    # Remove >
    book = rstrip(re.sub(r"^\>", " ", book, flags=re.MULTILINE))

    # Remove Headers # ## ### ...
    book = rstrip(re.sub(r"^(#){1,6}.{1,100}$", "", book, flags=re.MULTILINE))
    book = rstrip(
        re.sub(r"^(\*\*).{1,100}(\*\*).{0,100}$", "", book, flags=re.MULTILINE)
    )
    book = rstrip(
        re.sub(r"^((C|c)hapter|CHAPTER).{1,100}$", "", book, flags=re.MULTILINE)
    )

    # Remove ** Bold ** markers
    book = rstrip(re.sub(r"(?:\*\*)", " ", book))

    # Split into chapters and paragraphs
    chapters = re.compile(r"\n\n\n+", re.MULTILINE).split(book)
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
    chapters = [rstrip(re.sub(r"\ +", " ", chapter)).strip() for chapter in chapters]

    return chapters
