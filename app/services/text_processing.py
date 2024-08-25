import logging
import re

logger = logging.getLogger(__name__)


def remove_empty_lines(text: str) -> str:
    """Remove empty lines and lines that contain nothing but whitespaces from text.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The text with empty lines removed.
    """
    lines = text.split("\n")
    non_empty_lines = filter(lambda line: line.strip() != "", lines)

    return "\n".join(non_empty_lines)


def remove_excess_whitespace(text: str) -> str:
    """Removes excess whitespace from a text string.

    Args:
        text (str): The input text string to remove excess whitespace from.

    Returns:
        str: The text string with excess whitespace removed.
    """
    text = re.sub(r"\s+", " ", text)  ## remove general whitespace

    return text.strip()


def replace_special_characters(text: str, to_replace: dict[str, str]) -> str:
    """Replaces special characters and non breaking whitespaces from a string.

    Args:
        text (str): The input string containing emojis and emoticons.

    Returns:
        str: The input string without unicode characters and non breaking whitespaces.
    """
    for to_replace, replacement in to_replace.items():
        text = re.sub(to_replace, replacement, text)

    return text


def remove_non_utf8_characters(text: str) -> str:
    """Cleans a string from non-UTF-8 characters by replacing them with empty strings.

    Args:
        text (str): The input string to be cleaned.

    Returns:
        str: The cleaned string containing only UTF-8 characters.
    """
    return "".join(char for char in text if ord(char) < 256)


def remove_regex(text: str, regex: list[str]) -> str:
    """Remove substrings that match the specified regex from the text.

    Args:
    text (str): The input string to replace the expression.

    Returns:
        str: The input string without the substrings that match the regex.
    """

    for pattern in regex:
        text = re.sub(pattern, "", text).strip()

    return text


def remove_emojis(text: str) -> str:
    """Removes emojis from a string.

    Args:
        text (str): The input string containing emojis and emoticons.

    Returns:
        str: The input string with emojis and emoticons removed.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # alphanumeric
        "\U0001f926-\U0001f937"  # recent additions
        "\U00010000-\U0010ffff"  # supplementary
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    return text
