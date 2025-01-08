import os
from typing import Any, Dict

import yaml
from loguru import logger

from FishQueryML.utils.constants import PROJECT_ROOT


class ConfigReader:
    """A class to read and manage configuration from a YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ConfigReader with a path to the config file.

        Args:
            config_path (str): Path to the YAML config file
        """
        self.config_path = config_path
        self._config = None

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as config_file:
            self._config = yaml.safe_load(config_file)

        return self._config

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration. Loads it first if not already loaded.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if self._config is None:
            return self.load_config()
        return self._config

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific value from the configuration.

        Args:
            key (str): Key to look up
            default (Any): Default value if key not found

        Returns:
            Any: Value from config or default if not found
        """
        config = self.get_config()
        return config.get(key, default)


# Example usage:
if __name__ == "__main__":
    # Example config.yaml structure:
    """
    database:
      host: localhost
      port: 5432
      name: mydb

    api:
      url: https://api.example.com
      timeout: 30

    features:
      enable_cache: true
      max_retries: 3
    """

    # Create a config reader instance

    config_reader = ConfigReader(PROJECT_ROOT / "config.yaml")

    try:
        # Load and get full config
        config = config_reader.get_config()

        # Get specific values
        db_host = config_reader.get_value("platform", {}).get("url")
        api_url = config_reader.get_value("platform", {}).get("token")

        logger.info(f"Database host: {db_host}")
        logger.warning(f"API URL: {api_url}")

    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading config: {e}")
        logger.exception(e)
