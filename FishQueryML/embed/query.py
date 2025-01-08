from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from FishQueryML.utils.config_reader import ConfigReader
from FishQueryML.utils.constants import CONFIG_YAML


def search_rules(my_query: str, limit: int = 3):
    """Simple search function for fishing rules"""
    # Initialize client and model
    config = ConfigReader(CONFIG_YAML).get_config()
    client = QdrantClient(
        url=config['qdrant']['url'],
        port=config['qdrant']['port'],
    )
    model = SentenceTransformer('msmarco-bert-base-dot-v5')

    # Generate query embedding
    query_vector = model.encode(my_query)

    # Search
    results = client.search(
        collection_name="fishing_rules",
        query_vector=query_vector.tolist(),  # Convert numpy array to list
        limit=limit
    )

    # Print results
    print(f"\nSearch results for: {my_query}")
    print("-" * 50)

    for hit in results:
        print(f"\nScore: {hit.score:.3f}")
        print(f"Page: {hit.payload['page_number']}")
        print(f"Content: {hit.payload['content']}...")
        print("-" * 50)


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "mykiss",
        "Polyprion oxygeneios"
    ]

    for query in test_queries:
        search_rules(query)
