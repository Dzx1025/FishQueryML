import json
import re
import sys
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import time

from FishQueryML.utils.constants import DATA_DIR, CONFIG_YAML
from FishQueryML.utils.config_reader import ConfigReader


def clean_content(text: str) -> str:
    """Clean and normalize text content."""
    text = text.replace("<!-- image -->", "")
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters except alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', ' ', text)

    # Strip leading and trailing whitespace
    return text.strip()


def split_content(content: str) -> List[str]:
    """Split content into paragraphs by double newlines and clean each part."""
    # Split by double newlines and filter out empty strings
    parts = [part.strip() for part in content.split('\n\n')]
    return [part for part in parts if part]


def process_json_data(file_path: str) -> List[Tuple[int, str]]:
    """
    Process JSON data and split content into paragraphs.
    Returns a list of tuples containing (page_number, paragraph_content).
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    # Sort by page number to ensure consistent ordering
    for page_num in sorted(data.keys(), key=int):
        content = data[page_num]
        # Split content into paragraphs
        paragraphs = split_content(content)

        for paragraph in paragraphs:
            cleaned_paragraph = clean_content(paragraph)
            if cleaned_paragraph:  # Only add non-empty paragraphs
                processed_data.append((int(page_num), cleaned_paragraph))

    return processed_data


def wait_for_qdrant(client: QdrantClient, max_attempts: int = 5, wait_seconds: int = 2) -> bool:
    """Wait for Qdrant server to become available."""
    for attempt in range(max_attempts):
        try:
            # Try to get server info as a health check
            client.get_collections()
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} to connect to Qdrant failed: {str(e)}")
            if attempt < max_attempts - 1:
                print(f"Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
    return False


def embed_and_upload_to_qdrant(
        processed_data: List[Tuple[int, str]],
        collection_name: str,
        url: str,
        port: int
) -> None:
    """Embed and upload data to Qdrant with error handling."""
    try:
        # Initialize the embedding model
        model = SentenceTransformer('msmarco-bert-base-dot-v5')

        # Initialize Qdrant client
        client = QdrantClient(url=url, port=port)

        # Wait for Qdrant to become available
        if not wait_for_qdrant(client):
            raise ConnectionError(
                f"Could not connect to Qdrant server at {url}:{port}. "
                "Please ensure the server is running and accessible."
            )

        try:
            # Try to get collection info first
            client.get_collection(collection_name)
            # If we get here, collection exists, so delete it
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        except UnexpectedResponse:
            # Collection doesn't exist, which is fine
            pass

        # Create new collection
        print(f"Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )

        # Prepare data for upload
        points = []
        for idx, (page_num, content) in enumerate(processed_data):
            # Generate embedding for the content
            embedding = model.encode(content)

            # Create point
            point = models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'page_number': page_num,
                    'content': content,
                }
            )
            points.append(point)

        # Upload points in batches
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            print(f"Uploading batch {current_batch}/{total_batches}")
            client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"Successfully uploaded {len(points)} points to collection '{collection_name}'")

    except Exception as e:
        print(f"Error during embedding and upload process: {str(e)}", file=sys.stderr)
        raise


def main():
    try:
        # Configuration
        config = ConfigReader(CONFIG_YAML).get_config()
        json_file_path = DATA_DIR / "output" / "results.json"
        collection_name = "fishing_rules"
        qdrant_host = config['qdrant']['url']
        qdrant_port = config['qdrant']['port']

        # Process JSON data
        print("Processing JSON data...")
        processed_data = process_json_data(json_file_path)
        print(f"Processed {len(processed_data)} paragraphs across multiple pages")

        # Embed and upload to Qdrant
        print(f"Starting upload to Qdrant at {qdrant_host}:{qdrant_port}")
        embed_and_upload_to_qdrant(
            processed_data,
            collection_name=collection_name,
            url=qdrant_host,
            port=qdrant_port
        )

    except Exception as e:
        print(f"Error in main process: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
