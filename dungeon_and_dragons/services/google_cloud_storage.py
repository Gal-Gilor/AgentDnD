import logging
import os
from typing import Optional

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.api_core import exceptions
from google.cloud import storage
from google.oauth2 import service_account

from dungeon_and_dragons.core.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


def get_gstorage_client(
    service_account_credentials: str | None = None,
) -> storage.Client:
    """
    Get an authenticated Google Cloud Storage client.

    Args:
        service_account_credentials (str, optional):
            Path to the service account credentials file. Defaults to None.

    Returns:
        storage.Client: The authenticated Google Cloud Storage client.
    """
    if service_account_credentials or settings.GOOGLE_APPLICATION_CREDENTIALS:
        credentials_file = (
            service_account_credentials or settings.GOOGLE_APPLICATION_CREDENTIALS
        )

        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_file
            )
            client = storage.Client(credentials=credentials)
            logger.info(
                "Successfully connected to Google Cloud Storage using service account credentials."
            )
            return client

        except FileNotFoundError as e:
            logger.exception(
                f"Credentials file not found: {e}. Returning an unauthenticated client."
            )
            return storage.Client()

    return storage.Client()


def get_gstorage_bucket(bucket_name: str) -> storage.Bucket:
    """Retrieve a Google Cloud Storage bucket.

    This function attempts to retrieve a Google Cloud Storage bucket with
    the specified name. It uses the authenticated client returned by
    `get_gstorage_client`.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket to retrieve.

    Returns:
        storage.Bucket: The Google Cloud Storage bucket object.

    Raises:
        Unauthorized: If access to the bucket is unauthorized (HTTP 401).
        Forbidden: If access to the bucket is forbidden (HTTP 403).
        NotFound: If the bucket is not found (HTTP 404).
        HTTPException: For any other errors encountered while trying to access
        the bucket (HTTP 500).
    """
    try:
        client: storage.Client = get_gstorage_client()
        bucket: storage.Bucket = client.get_bucket(bucket_name)
        logger.info(f"Successfully connected to {bucket_name}.")
        return bucket

    except exceptions.Unauthorized as e:
        logger.error(f"Unauthorized access to {bucket_name}: {e}.")
        raise HTTPException(
            status_code=401, detail=f"Unauthorized access to {bucket_name}: {e}."
        )

    except exceptions.Forbidden as e:
        logger.error(f"Unable to access {bucket_name}: {e}.")
        raise HTTPException(
            status_code=403, detail=f"Unable to access {bucket_name}: {e}."
        )

    except exceptions.NotFound as e:
        logger.error(f"Bucket {bucket_name} not found: {e}.")
        raise HTTPException(
            status_code=404, detail=f"Bucket {bucket_name} not found: {e}."
        )

    except Exception as e:
        logger.error(f"Unknown error connecting to {bucket_name}: {e}.", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Unknown error connecting to {bucket_name}: {e}."
        )


def upload_to_bucket(file: UploadFile, bucket_name: str, destination_blob: str) -> str:
    """Uploads a file to a specified Google Cloud Storage bucket.

    Args:
        file (UploadFile): The file to be uploaded.
            This should be an instance of FastAPI's `UploadFile`.
        bucket_name (str): The name of the Google Cloud Storage bucket to be uploaded.
        destination_blob (str): The destination path and file name in the bucket.
            This is where the file will be stored within the bucket.

    Returns:
        str: The gsutil URI of the uploaded file in the format
            `gs://bucket_name/destination_blob`.

    Raises:
        google.api_core.exceptions.Conflict: If there is a conflict during the upload,
            such as a file with the same name already existing in the bucket.
        Exception: For any other errors encountered during the upload process.
            The specific error will be logged, and an exception will be raised.
    """
    try:
        bucket: storage.Bucket = get_gstorage_bucket(bucket_name)
        blob: storage.Blob = bucket.blob(destination_blob)
        blob.upload_from_file(file.file)
        logger.info(
            f"File '{file.filename}' uploaded to '{destination_blob}' in bucket '{bucket_name}'."
        )
        return f"gs://{bucket_name}/{destination_blob}"

    except exceptions.Conflict as e:
        logger.error(
            f"Conflict uploading '{file.filename}' to '{bucket_name}': {e}.",
        )
        raise HTTPException(
            status_code=409,
            detail=f"Conflict uploading '{file.filename}' to '{bucket_name}': {e}.",
        )

    except Exception as e:  # Catch broader exception for Google API errors
        logger.error(
            f"Error uploading '{file.filename}' to '{bucket_name}': {e}.",
            exc_info=True,
        )
        return HTTPException(
            status_code=500,
            detail=f"Error uploading '{file.filename}' to '{bucket_name}': {e}.",
        )


# def download_from_bucket(filename: str, destination_path: Optional[str] = ""):
#     """
#     Downloads a file from the specified bucket in Google Cloud Storage.

#     Args:
#         filename (str): The name of the file in the bucket.
#         destination_path (str, optional): The local path where the file will be saved. If not provided,
#                                             it will be saved in the current working directory.
#     """
#     try:
#         bucket = get_bucket(bucket_name)
#         blob = bucket.blob(filename)

#         if not destination_path:
#             destination_path = os.path.join(os.getcwd(), filename)

#         blob.download_to_filename(destination_path)
#         logger.info(f"{filename} successfully downloaded to {destination_path}.")

#     except Exception as e:
#         exception = extract_error_from_exceptions(str(e))
#         logger.exception(f"Error downloading {filename} from bucket: {exception}")
