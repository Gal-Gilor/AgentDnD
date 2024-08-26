import os
from unittest.mock import MagicMock, patch

import google.oauth2.service_account
import pytest
from fastapi import HTTPException, UploadFile
from google.api_core import exceptions
from google.cloud import storage

from app.services.google_cloud_storage import (
    get_gstorage_bucket,
    get_gstorage_client,
    upload_to_bucket,
)


@patch("google.cloud.storage.Client")
@patch("google.oauth2.service_account.Credentials.from_service_account_file")
def test_get_gstorage_client_with_valid_credentials(
    mock_from_service_account_file, mock_storage_client, mock_settings
):
    mock_credentials = MagicMock()
    mock_storage_client_instance = MagicMock()

    # Setup mock return values
    mock_from_service_account_file.return_value = mock_credentials
    mock_storage_client.return_value = mock_storage_client_instance

    # Call the function with a valid credentials file
    credentials_path = "/path/to/credentials.json"
    client = get_gstorage_client(service_account_credentials=credentials_path)

    # Assert that the client was created with the mock credentials
    mock_storage_client.assert_called_with(credentials=mock_credentials)


@patch("google.cloud.storage.Client")
@patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": ""}, clear=True)
def test_get_gstorage_client_without_credentials(mock_storage_client):
    # Ensure the mock for Client does not use any credentials
    mock_storage_client_instance = MagicMock()
    mock_storage_client.return_value = mock_storage_client_instance

    # Call the function which should use the mocked Client
    client = get_gstorage_client()

    # Ensure that the Client was created with no credentials
    mock_storage_client.assert_called_once_with()


@patch("google.cloud.storage.Client")
@patch(
    "google.oauth2.service_account.Credentials.from_service_account_file",
    side_effect=FileNotFoundError,
)
def test_get_gstorage_client_with_file_not_found(
    mock_from_service_account_file, mock_storage_client, mock_settings
):
    mock_storage_client_instance = MagicMock()

    # Setup mock return value
    mock_storage_client.return_value = mock_storage_client_instance

    # Call the function with a valid credentials file path
    credentials_path = "/path/to/credentials.json"
    client = get_gstorage_client(service_account_credentials=credentials_path)

    # Assert that the client was created without credentials due to file not found
    mock_storage_client.assert_called_with()


def test_get_gstorage_bucket_success(mocker, mock_settings, mock_storage_client):
    mock_bucket = MagicMock()
    mock_storage_client.return_value.get_bucket.return_value = mock_bucket
    mocker.patch(
        "app.services.google_cloud_storage.get_gstorage_client",
        return_value=mock_storage_client.return_value,
    )

    bucket_name = "my-bucket"
    bucket = get_gstorage_bucket(bucket_name)

    mock_storage_client.return_value.get_bucket.assert_called_once_with(bucket_name)
    assert bucket == mock_bucket


def test_get_gstorage_bucket_not_found(mocker, mock_settings, mock_storage_client):
    # Mock the behavior of get_bucket to raise NotFound exception
    mock_client = mock_storage_client.return_value
    mock_client.get_bucket.side_effect = exceptions.NotFound("Bucket not found")

    # Patch get_gstorage_client to return the mocked client
    mocker.patch(
        "app.services.google_cloud_storage.get_gstorage_client",
        return_value=mock_client,
    )

    with pytest.raises(HTTPException) as excinfo:
        get_gstorage_bucket("non-existent-bucket")

    assert excinfo.value.status_code == 404
    assert "Bucket non-existent-bucket not found" in str(excinfo.value.detail)


def test_upload_to_bucket_success(mocker, mock_settings, mock_storage_client):
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.upload_from_file = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.get_bucket.return_value = mock_bucket
    mocker.patch(
        "app.services.google_cloud_storage.get_gstorage_client",
        return_value=mock_storage_client.return_value,
    )

    file = MagicMock(spec=UploadFile)
    file.file = MagicMock()
    file.filename = "test_file.txt"
    bucket_name = "my-bucket"
    destination_blob = "uploads/test_file.txt"

    result = upload_to_bucket(file, bucket_name, destination_blob)

    mock_bucket.blob.assert_called_once_with(destination_blob)
    mock_blob.upload_from_file.assert_called_once_with(file.file)
    assert result == f"gs://{bucket_name}/{destination_blob}"


def test_upload_to_bucket_conflict(mocker, mock_settings, mock_storage_client):
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.upload_from_file.side_effect = exceptions.Conflict("Conflict")
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client.return_value.get_bucket.return_value = mock_bucket
    mocker.patch(
        "app.services.google_cloud_storage.get_gstorage_client",
        return_value=mock_storage_client.return_value,
    )

    file = MagicMock(spec=UploadFile)
    file.file = MagicMock()
    file.filename = "test_file.txt"
    bucket_name = "my-bucket"
    destination_blob = "uploads/test_file.txt"

    with pytest.raises(HTTPException) as excinfo:
        upload_to_bucket(file, bucket_name, destination_blob)

    assert "Conflict uploading" in str(excinfo.value.detail)
