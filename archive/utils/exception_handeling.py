import os
import re

from .configurations import config_from_file


def extract_pattern_from_text(text: str, pattern_expression: str):
    """
    Extracts patterns from a given text using a regular expression pattern.

    Args:
        text (str): The text to search for patterns.
        pattern_expression (str): The regular expression pattern to search for.

    Returns:
        list: A list of matches found in the text.
    """
    try:
        return re.findall(pattern_expression, text)

    except Exception as e:
        # Log or handle the exception
        return []


def extract_error_from_exceptions(exception_text: str):
    """
    Extracts error information from exceptions in the provided text using given patterns.

    Args:
        text (str): The text containing exception information.
        patterns (dict): A dictionary where keys are names of error components and values are regular expression patterns.
            for example, {"code": '"code": (\d+)')

    Returns:
        dict: A dictionary containing error components extracted from the text based on the provided patterns.
    """
    config = config_from_file(os.environ["DOWNLOADS_CONFIG_PATH"])
    patterns = config["patterns"]
    error = {}
    try:
        # Use list comprehension to populate the error dictionary
        error = {
            name: extract_pattern_from_text(exception_text, expression)
            for name, expression in patterns.items()
        }

        if not error.values():
            return exception_text

    except Exception as e:
        # Log or handle the exception
        return exception_text

    return error
