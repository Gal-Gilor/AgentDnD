import logging

from fastapi import APIRouter, HTTPException, Query
from google.api_core.exceptions import (
    BadGateway,
    BadRequest,
    InternalServerError,
    TooManyRequests,
    Unauthorized,
)
from models.instructor_models.generate_adventure import ImprovedAdventure
from services.generate_adventure import generate_adventure

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
adventure_router = APIRouter()


@adventure_router.get("/generate_adventure", response_model=ImprovedAdventure)
async def generate_adventure_endpoint(
    specifications: str = Query(
        ..., description="The user provided specifications for the adventure."
    )
):
    """Generate a custom Dungeons & Dragons adventure based on user specifications.

    Args:
        specifications (str): A string detailing the desired adventure, including setting,
            plot, characters, and themes.

    Returns:
        ImprovedAdventure: The generated adventure with a complete story framework,
        encounters, NPCs, and locations.

    Raises:
        HTTPException: Raised for:
            - 400 Bad Request: Invalid specifications or request.
            - 500 Internal Server Error: Service errors or unexpected issues.
            - 429 Too Many Requests: Overloaded service.
            - 401 Unauthorized: Authorization issues.
    """
    try:
        adventure = generate_adventure(specifications)
        return adventure

    except ValueError as e:
        logger.error(f"ValidationError: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

    except (InternalServerError, BadGateway) as e:
        logger.error(f"API 500 Error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except (BadRequest, TooManyRequests, Unauthorized) as e:
        logger.error(f"API 400 Error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
