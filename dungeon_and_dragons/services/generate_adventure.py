import logging
from typing import Optional

import google.generativeai as genai
import instructor
from core.settings import Settings
from google.api_core.exceptions import (
    BadGateway,
    BadRequest,
    InternalServerError,
    TooManyRequests,
    Unauthorized,
)
from models.instructor_models.generate_adventure import (
    CreateAdventure,
    ImprovedAdventure,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_adventure(specifications: str) -> ImprovedAdventure:
    """Generates a detailed and engaging D&D adventure.

    Args:
        specifications: A string describing the desired adventure.

    Returns:
        ImprovedAdventure: An object containing the generated adventure.

    Raises:
        ValueError: If there is a validation error.
        InternalServerError: If the API returns a 500 error.
        BadGateway: If the API returns a 502 error.
        BadRequest: If the API returns a 400 error.
        TooManyRequests: If the API returns a 429 error.
        Unauthorized: If the API returns a 401 error.
        Exception: If an unexpected error occurs.
    """
    try:
        # Initialize the Instructor client with the Gemini model
        client = instructor.from_gemini(
            client=genai.GenerativeModel("gemini-1.5-flash-001"),
            mode=instructor.Mode.GEMINI_JSON,
        )

        # Generate a basic adventure using the specifications
        response: CreateAdventure = client.create(
            response_model=CreateAdventure,
            messages=[{"role": "user", "content": specifications}],
        )

        # Log the generated adventure title
        adventure_title = response.model_dump().get("title")
        logger.info(
            "Finished writing a rough draft of the adventure. "
            f"Title: {adventure_title}"
        )

        # Prepare instructions to improve the adventure
        tweaking_instruction = (
            "As an experienced Dungeon Master, your goal is to significantly enhance the following D&D adventure. "
            "Transform it into a more immersive, detailed, and engaging story. Each description should be at least "
            "3 paragraphs long, using vivid and evocative language to fully bring the world to life. Ensure that every "
            "aspect of the adventure is rich in detail and avoids generic tropes.\n"
            "If the adventure involves a mystery, enhance it with additional clues and improve logical reasoning to "
            "create a compelling and solvable puzzle. Ensure that clues are well-integrated into the narrative.\n"
            "Include at least 6 diverse encounters that are thoroughly described and align with the adventure’s theme. "
            "Each encounter should offer meaningful challenges and rewards, contributing to the story's progression.\n"
            f"**Adventure**:\n {response.model_dump_json()}"
        )

        # Improve the adventure with detailed descriptions and encounters
        logger.info("Ensuring the generated adventure is cohesive and detailed...")
        adventure: ImprovedAdventure = client.create(
            response_model=ImprovedAdventure,
            messages=[{"role": "user", "content": tweaking_instruction}],
        )
        logger.info("Successfully generated a cohesive and detailed adventure.")

        return adventure

    except ValidationError as e:
        logger.error(f"ValidationError: {e}", exc_info=True)
        raise ValueError(f"ValidationError: {e.errors()}")

    except (InternalServerError, BadGateway) as e:
        logger.error(f"API 500 Error occurred: {e}", exc_info=True)
        raise e

    except (BadRequest, TooManyRequests, Unauthorized) as e:
        logger.error(f"API 400 Error occurred: {e}", exc_info=True)
        raise e

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise Exception("An unexpected error occurred.")
