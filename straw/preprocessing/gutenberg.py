import re
from typing import List
from .utils import filter_pargraphs, rstrip


def process_gutenberg(raw_book) -> List[str]:
    """
    Processes a gutenberg file and returns list of cleaned paragraphs.
    """
    # Clean encoding errors
    # Replace invalid quotes with valid ones
    book = re.sub(r"â[¦]", "'", raw_book)
    book = re.sub(r"Ã©", "é", book)

    # Remove illustrations
    book = re.sub(r"\[.*(\n.*){0,10}\]", "", book)

    # italics
    book = rstrip(re.sub(r"_(.*?)_", r" \1 ", book))
    book = rstrip(re.sub(r"\[(.*?)\]", r" \1 ", book))

    # Remove >
    book = rstrip(re.sub(r"^\>", " ", book, flags=re.MULTILINE))

    # Remove license and header
    endidx = next(
        re.finditer(
            r"\n.*(\*\*\*.*END OF.*\*\*\*|End.*Project Guten.*|\*THE END.\*)",
            book,
            re.MULTILINE,
        ),
        None,
    )
    book = book[: endidx.start()] if endidx else book

    def remove_header(book):
        startidx = next(
            re.finditer(
                r"("
                "\*\*\*.*START OF.*\*\*\*"
                "|Produced by.*"
                "|This etext was produced.*"
                "|E\-text prepared by .*"
                "|.*Transcribed from.*"
                "|.*Project Gutenberg's Etext of.*"
                ")(\n.*){0,3}\n\n",
                book,
                re.MULTILINE,
            ),
            None,
        )
        if startidx and startidx.end() < 10000:
            return remove_header(book[startidx.end() :])
        return book

    book = remove_header(book)

    # Split chapters
    chapters = re.compile("\n\n\n+|\*\ *\*\ *\*\ *\*\ *\*\ *", re.MULTILINE).split(book)

    # Split chapters into paragraphs
    chapters = list(map(re.compile("\n", re.MULTILINE).split, chapters))

    # Fix paragraph formatting by removing word wrap.
    chapters = [
        [
            re.sub(r"(?!.{50,})\n(?=[^\n])", " ", paragraph).strip()
            for paragraph in chapter
            if paragraph.strip() and not paragraph.isupper()
        ]
        for chapter in chapters
    ]

    # Join paragraphs
    chapters = [rstrip("\n".join(chapter)).strip() for chapter in chapters]

    # Remove lists
    chapters = [
        rstrip(
            re.sub(
                r"(^((C|c)hapter|CHAPTER)*\ *[\dIVX]+.{0,200}\n+){3,}",
                "",
                chapter,
                flags=re.MULTILINE,
            )
        )
        for chapter in chapters
    ]

    # Remove redundant chapters (ratio of non alphabetics,
    # or shorter chapters maybe, contains any of the keywords)
    chapters = [chapter for chapter in chapters if filter_pargraphs(chapter)]

    # Clean whitespaces
    chapters = [rstrip(re.sub(r"\ +", " ", chapter)).strip() for chapter in chapters]

    return chapters
