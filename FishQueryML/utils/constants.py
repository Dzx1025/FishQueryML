from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent.parent

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

CONFIG_YAML = PROJECT_ROOT / "config.yaml"

if not CONFIG_YAML.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_YAML}")

DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)
