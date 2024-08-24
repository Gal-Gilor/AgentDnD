import logging
import logging.config

import yaml


def config_from_file(filename: str) -> dict:
    """Load configuration data from a YAML file.

    Args:
        filename (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration data.
    """
    try:
        with open(filename) as file:
            return yaml.safe_load(file)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading config file '{filename}': {e}")

    except Exception as e:
        raise Exception(f"Unable to load config file '{filename}': {e}")


def logger_from_file(filename: str) -> logging.Logger:
    """Configure logger using YAML configuration file.

    Args:
        filename (str): The path to the YAML logging configuration file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    try:
        # Open and load the configuration
        config = config_from_file(filename)
        if not config:
            raise ValueError(
                f"Invalid or empty logging configuration in file '{filename}'"
            )

        # Configure the logger
        logging.config.dictConfig(config)

        return logging.getLogger(__name__)

    except ValueError as e:
        raise ValueError(f"Invalid logging configuration in file '{filename}': {e}")

    except Exception as e:
        raise Exception(f"Unable to load logger config file '{filename}': {e}")


def logger_from_file(filename: str) -> logging.Logger:
    """Configure logger using YAML configuration file."""
    try:
        # open and load the configuration
        config = config_from_file(filename)
        logging.config.dictConfig(config)

        return logging.getLogger(__name__)

    except Exception as e:
        raise Exception(f"Unable to load logger config file '{filename}': {e}")
