import logging
import logging.config

import yaml


def config_from_file(filename: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(filename) as file:
            return yaml.safe_load(file)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading config file '{filename}': {e}")

    except Exception as e:
        raise Exception(f"Unable to load config file '{filename}': {e}")


def logger_from_file(filename: str) -> logging.Logger:
    """Configure logger using YAML configuration file."""
    try:
        # open and load the configuration
        config = config_from_file(filename)
        logging.config.dictConfig(config)

        return logging.getLogger(__name__)

    except Exception as e:
        raise Exception(f"Unable to load logger config file '{filename}': {e}")
