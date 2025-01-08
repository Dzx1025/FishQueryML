import time
from loguru import logger


class timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.info(f"Starting {self.name}")
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        logger.info(f"Finished {self.name}")
        logger.info(f"{self.name} took {self.end - self.start} seconds")


if __name__ == "__main__":
    with timer("Sleep"):
        time.sleep(1)
