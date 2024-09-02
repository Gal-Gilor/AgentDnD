import pytest


@pytest.fixture
def sample_text() -> str:
    return "  This is a  sample\ntext with\n\n  empty  lines and  \n\n extra spaces."


@pytest.fixture
def to_replace() -> dict[str, str]:
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


@pytest.fixture
def mock_settings(mocker):
    mock_settings = mocker.patch(
        "dungeon_and_dragons.core.settings.Settings", autospec=True
    )
    return mock_settings


@pytest.fixture
def mock_storage_client(mocker):

    # Mock the Google Cloud Storage Client
    mock_client = mocker.MagicMock()
    mocker.patch(
        "dungeon_and_dragons.services.google_cloud_storage.get_gstorage_client",
        return_value=mock_client,
    )

    return mock_client
