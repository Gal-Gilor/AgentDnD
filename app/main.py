import logging

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.google_cloud_storage import upload_to_bucket

app = FastAPI()
logger = logging.getLogger(__name__)


@app.post("/upload")
async def upload_file_to_gcs(
    bucket_name: str, destination_blob: str, file: UploadFile = File(...)
):
    """
    Endpoint to upload a file to Google Cloud Storage.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        destination_blob (str): The destination path and file name in the bucket.
        file (UploadFile): The file to be uploaded.

    Returns:
        JSONResponse: A JSON response with the gsutil URI or error details.
    """
    try:
        result = upload_to_bucket(file, bucket_name, destination_blob)
        if isinstance(result, JSONResponse):
            raise HTTPException(
                status_code=result.status_code, detail=result.content["detail"]
            )
        return result

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
