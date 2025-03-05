import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
from loguru import logger
# For Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from FishQueryML.utils.config_reader import ConfigReader
from FishQueryML.utils.constants import DATA_DIR, CONFIG_YAML

# Load configuration
config = ConfigReader(CONFIG_YAML).get_config()

# Ensure output directory exists
OUTPUT_DIR = DATA_DIR / "output"
if not OUTPUT_DIR.exists():
    logger.error(f"Output directory does not exist: {OUTPUT_DIR}")
    raise NotADirectoryError(f"Output directory does not exist: {OUTPUT_DIR}")

json_file = DATA_DIR / "output" / "origin_chunk.json"

if not json_file:
    logger.error(f"No JSON files found to process in: {OUTPUT_DIR}")
    raise FileNotFoundError(f"No JSON files found to process in: {OUTPUT_DIR}")

# Qdrant configuration
qdrant_url = config['qdrant']['url']
collection_name = config['qdrant']['collection_name']
vector_size = 768  # Nomic embeddings dimensionality

# Nomic API configuration
NOMIC_TOKEN = config['nomic']['token']
NOMIC_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = config['nomic']['model']
TASK_TYPE = "search_document"

# Batch sizes for processing
EMBEDDING_BATCH_SIZE = 32
UPLOAD_BATCH_SIZE = 100
RECREATE_COLLECTION = False


def init_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False):
    """Initialize Qdrant collection if it doesn't exist, or recreate if specified."""
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name in collection_names:
            if recreate:
                logger.info(f"Recreating Qdrant collection: {collection_name}")
                client.delete_collection(collection_name=collection_name)
            else:
                logger.info(f"Qdrant collection already exists: {collection_name}")
                return

        logger.info(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        logger.success(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise


def load_json_documents(file_path: Path) -> List[Dict[str, Any]]:
    """Load document chunks from a JSON file."""
    logger.info(f"Loading documents from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.success(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents from {file_path}: {e}")
        raise


def get_nomic_embeddings(texts: List[str], batch_size: int) -> List[List[float]]:
    """Get embeddings using Nomic API with batching and progress tracking."""
    logger.info(f"Generating embeddings for {len(texts)} documents using Nomic API")

    headers = {
        "Authorization": f"Bearer {NOMIC_TOKEN}",
        "Content-Type": "application/json"
    }

    all_embeddings = []

    # Process in batches with tqdm progress bar
    with tqdm(total=len(texts), desc="Generating embeddings", unit="docs") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:min(i + batch_size, len(texts))]

            # Prepare payload for Nomic API
            payload = {
                "model": NOMIC_MODEL,
                "texts": batch_texts,
                "task_type": TASK_TYPE,
                "dimensionality": vector_size
            }

            # Make API request with retry logic
            max_retries = 3
            retry_count = 0
            success = False

            while not success and retry_count < max_retries:
                try:
                    response = requests.post(NOMIC_URL, headers=headers, json=payload)

                    if response.status_code == 200:
                        embeddings = response.json()["embeddings"]
                        all_embeddings.extend(embeddings)
                        success = True
                    elif response.status_code == 429:  # Rate limit
                        retry_count += 1
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.warning(f"Rate limited by Nomic API. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Nomic API error: {response.status_code} - {response.text}")
                        raise Exception(f"Failed to get embeddings: {response.text}")
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to get embeddings after {max_retries} retries: {e}")
                        raise
                    wait_time = 2 ** retry_count
                    logger.warning(f"Error connecting to Nomic API. Retrying in {wait_time} seconds... Error: {e}")
                    time.sleep(wait_time)

            pbar.update(len(batch_texts))

    logger.success(f"Generated embeddings for {len(texts)} documents")
    return all_embeddings


def upload_to_qdrant(client: QdrantClient, collection_name: str, documents: List[Dict[str, Any]],
                     embeddings: List[List[float]], batch_size: int):
    """Upload documents with embeddings to Qdrant."""
    logger.info(f"Uploading {len(documents)} documents to Qdrant collection: {collection_name}")

    # Prepare points for Qdrant
    points = []
    for i, doc in enumerate(documents):
        # Create a deterministic ID based on the document content for deduplication
        content = doc["page_content"].encode('utf-8')
        source = str(doc["metadata"].get("source", "")).encode('utf-8')
        point_id = hashlib.md5(content + source).hexdigest()

        # Create a Qdrant point
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"]
                }
            )
        )

    # Upload in batches with progress bar
    with tqdm(total=len(points), desc="Uploading to Qdrant", unit="docs") as pbar:
        for i in range(0, len(points), batch_size):
            batch_points = points[i:min(i + batch_size, len(points))]
            client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            pbar.update(len(batch_points))

    logger.success(f"Uploaded {len(documents)} documents to Qdrant")


def process_file(file_path: Path, client: QdrantClient):
    """Process a single JSON file, embed the documents and upload to Qdrant."""
    logger.info(f"Processing file: {file_path}")

    try:
        # Load documents from JSON file
        documents = load_json_documents(file_path)

        # Extract texts for embedding
        texts = [doc["page_content"] for doc in documents]

        # Generate embeddings
        embeddings = get_nomic_embeddings(texts, EMBEDDING_BATCH_SIZE)

        # Upload to Qdrant
        upload_to_qdrant(client, collection_name, documents, embeddings, UPLOAD_BATCH_SIZE)

        logger.success(f"Successfully processed file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        raise


def main():
    """Main function to embed documents and store in Qdrant."""
    try:
        logger.info("Starting document embedding process")

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at: {qdrant_url}")
        client = QdrantClient(url=qdrant_url)

        # Initialize collection
        init_qdrant_collection(client, collection_name, vector_size, RECREATE_COLLECTION)

        process_file(json_file, client)
        logger.success("Successfully embedded all documents into Qdrant")

    except Exception as e:
        logger.error(f"Embedding process failed: {e}")
        raise


if __name__ == "__main__":
    main()
