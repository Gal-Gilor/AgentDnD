import pytest

from app.services.text_processing import (
    remove_emojis,
    remove_empty_lines,
    remove_excess_whitespace,
    remove_non_utf8_characters,
    remove_regex,
    replace_special_characters,
)


@pytest.mark.parametrize(
    "expected_output",
    ["  This is a  sample\ntext with\n  empty  lines and  \n extra spaces."],
)
def test_remove_empty_lines(sample_text: str, expected_output: str):
    assert remove_empty_lines(sample_text) == expected_output


@pytest.mark.parametrize(
    "expected_output",
    [
        "This is a sample text with empty lines and extra spaces.",
    ],
)
def test_remove_excess_whitespace(sample_text: str, expected_output: str):
    assert remove_excess_whitespace(sample_text) == expected_output


@pytest.mark.parametrize(
    "text, to_replace, expected_output",
    [
        (
            "This is a sample text with @<> characters.",
            {"@": "&amp;", "<": "&lt;", ">": "&gt;"},
            "This is a sample text with &amp;&lt;&gt; characters.",
        ),
        (
            "This is a sample text without special characters.",
            {},
            "This is a sample text without special characters.",
        ),
    ],
)
def test_replace_special_characters(
    text, to_replace: dict[str, str], expected_output: str
):
    assert replace_special_characters(text, to_replace) == expected_output


@pytest.mark.parametrize(
    "text, expected_output",
    [
        (
            "This is a sample text with non-UTF-8 characters: \u0300\u0301",
            "This is a sample text with non-UTF-8 characters: ",
        ),
        (
            "This is a sample text without non-UTF-8 characters.",
            "This is a sample text without non-UTF-8 characters.",
        ),
    ],
)
def test_remove_non_utf8_characters(text, expected_output: str):
    input_text = "This is a sample text with non-UTF-8 characters: \u0300\u0301"
    assert remove_non_utf8_characters(text) == expected_output


@pytest.mark.parametrize(
    "text, regex, expected_output",
    [
        (
            "This is a sample text with some regex patterns: [123] and (abc).",
            ["\[123\]", "\(abc\)"],
            "This is a sample text with some regex patterns:  and .",
        ),
        (
            "This is a sample text without regex patterns.",
            [],
            "This is a sample text without regex patterns.",
        ),
    ],
)
def test_remove_regex(text, regex: list[str], expected_output: str):
    input_text = "This is a sample text with some regex patterns: [123] and (abc)."
    assert remove_regex(text, regex) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        (
            "This is a sample text with emojis: 😊😂🤣",
            "This is a sample text with emojis: ",
        ),
        (
            "This is a sample text without emojis.",
            "This is a sample text without emojis.",
        ),
    ],
)
def test_remove_emojis(input_text: str, expected_output: str):
    assert remove_emojis(input_text) == expected_output
