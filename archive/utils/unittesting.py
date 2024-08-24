import unittest

from utils.configurations import config_from_file


def evaluate_config(config: dict):
    """Inspect the loaded configuration for completeness and correctness.

    Args:
        config (dict): The configuration dictionary loaded from the file.

    Raises:
        ValueError: If the configuration is missing required sections or fields.
    """
    required_sections = ["tokenizer", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Configuration is missing '{section}' section.")

    required_fields = ["model_id"]
    for section in required_sections:
        for field in required_fields:
            if field not in config[section]:
                raise ValueError(f"'{field}' field is missing in '{section}' section.")


class TestConfig(unittest.TestCase):
    def test_valid_config(self):
        """Test a valid configuration."""
        config = config_from_file("config.yaml")
        self.assertIsNone(evaluate_config(config))

    def test_missing_section(self):
        """Test configuration with a missing section."""
        config = {"model": {"model_id": "model_name"}}
        with self.assertRaises(ValueError):
            evaluate_config(config)

    def test_missing_field(self):
        """Test configuration with a missing field."""
        config = {"tokenizer": {}, "model": {"model_id": "model_name"}}
        with self.assertRaises(ValueError):
            evaluate_config(config)


if __name__ == "__main__":
    unittest.main()
