import pytest


@pytest.fixture
def sample_text() -> str:
    return "  This is a  sample\ntext with\n\n  empty  lines and  \n\n extra spaces."


@pytest.fixture
def to_replace() -> dict[str, str]:
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
