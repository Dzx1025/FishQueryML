import os
import json
import PyPDF2
from pathlib import Path
from typing import Dict, Any, List, Union
from loguru import logger
from tqdm import tqdm

from FishQueryML.utils.constants import DATA_DIR
from FishQueryML.utils.config_reader import ConfigReader
from FishQueryML.utils.constants import CONFIG_YAML

# Load configuration
config = ConfigReader(CONFIG_YAML).get_config()
chunk_size = config.get('chunk_size', 1000)
chunk_overlap = config.get('chunk_overlap', 200)

# Ensure output directory exists
OUTPUT_DIR = DATA_DIR / "output"
if not OUTPUT_DIR.exists():
    logger.error(f"Output directory does not exist: {OUTPUT_DIR}")
    raise NotADirectoryError(f"Output directory does not exist: {OUTPUT_DIR}")


def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into chunks using a recursive approach similar to Langchain's RecursiveCharacterTextSplitter.
    """
    # Define the separators to use, in order of priority
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _split_text_recursive(text, separators, chunk_size, chunk_overlap)


def _split_text_recursive(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Helper function to recursively split text."""
    # If text is already small enough, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    # If we have no more separators, use character chunking
    if not separators:
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:min(i + chunk_size, len(text))]
            chunks.append(chunk)
        return chunks

    # Use the first separator
    separator = separators[0]

    # If separator is empty, use character chunking
    if separator == "":
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:min(i + chunk_size, len(text))]
            chunks.append(chunk)
        return chunks

    # Split the text by the separator
    splits = text.split(separator)

    # If we only got one split and the text is too large, try the next separator
    if len(splits) == 1 and len(text) > chunk_size:
        return _split_text_recursive(text, separators[1:], chunk_size, chunk_overlap)

    # Initialize chunks
    chunks = []
    current_chunk = []
    current_length = 0

    # Process each split
    for split in splits:
        # Calculate the length if we add this split
        split_length = len(split)
        if current_chunk:  # Add separator length if not the first split
            split_length += len(separator)

        # If adding this split would exceed chunk_size, finalize the current chunk
        if current_length + split_length > chunk_size and current_chunk:
            # Join the current chunks with the separator
            joined_chunk = separator.join(current_chunk)
            chunks.append(joined_chunk)

            # Handle overlap
            if chunk_overlap > 0 and len(current_chunk) > 0:
                # Keep the last split for overlap
                current_chunk = [current_chunk[-1]]
                current_length = len(current_chunk[-1])
            else:
                current_chunk = []
                current_length = 0

        # Add this split to the current chunk
        current_chunk.append(split)
        current_length += split_length

        # If the current chunk is now too large, recursively split it
        if current_length > chunk_size:
            # Join and split recursively
            joined_text = separator.join(current_chunk)
            recursive_chunks = _split_text_recursive(joined_text, separators[1:], chunk_size, chunk_overlap)
            chunks.extend(recursive_chunks)
            # Reset current chunk
            current_chunk = []
            current_length = 0

    # Don't forget the last chunk
    if current_chunk:
        joined_chunk = separator.join(current_chunk)
        # If it's too large, split recursively
        if len(joined_chunk) > chunk_size:
            recursive_chunks = _split_text_recursive(joined_chunk, separators[1:], chunk_size, chunk_overlap)
            chunks.extend(recursive_chunks)
        else:
            chunks.append(joined_chunk)

    return chunks


def load_and_split_pdf(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Load a PDF file, extract text page by page, and split into chunks."""
    logger.info(f"Reading and splitting PDF file: {file_path}")
    documents = []

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            for i in tqdm(range(total_pages), desc="Reading and splitting PDF pages"):
                page = pdf_reader.pages[i]
                text = page.extract_text()

                if text.strip():  # Only process non-empty pages
                    # Split the text into chunks
                    chunks = split_text_recursive(text, chunk_size, chunk_overlap)

                    # Add each chunk as a separate document
                    for j, chunk in enumerate(chunks):
                        documents.append({
                            'page_content': chunk,
                            'metadata': {
                                'page': i,
                                'chunk': j,
                                'total_pages': total_pages,
                                'source': str(file_path)
                            }
                        })

        logger.success(f"PDF file processing completed: created {len(documents)} chunks from {total_pages} pages")
        return documents
    except Exception as e:
        logger.error(f"Failed to read and split PDF file: {e}")
        raise


def load_and_split_text(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Load a text file and split it into chunks."""
    logger.info(f"Reading and splitting text file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Split the text into chunks
        chunks = split_text_recursive(text, chunk_size, chunk_overlap)

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'page_content': chunk,
                'metadata': {
                    'chunk': i,
                    'source': str(file_path)
                }
            })

        logger.success(f"Text file processing completed: created {len(documents)} chunks from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to read and split text file: {e}")
        raise


def process_file_with_splitting(file_path: Union[str, Path], chunk_size: int, chunk_overlap: int) -> Path:
    """Process a file, split content into chunks, and store to output directory."""
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Get file extension
    ext = file_path.suffix.lower()

    # Create output filename
    output_file = OUTPUT_DIR / f"{file_path.stem}_chunk.json"

    logger.info(f"Starting to process and split file: {file_path}")

    # Select appropriate loader based on file extension
    if ext == ".pdf":
        documents = load_and_split_pdf(file_path, chunk_size, chunk_overlap)
    else:  # Default to text loader
        documents = load_and_split_text(file_path, chunk_size, chunk_overlap)

    # Save documents as JSON format
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.success(f"File processing and splitting completed, output saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to save processing results: {e}")
        raise


def process_directory_with_splitting(dir_path: Union[str, Path], chunk_size: int, chunk_overlap: int):
    """Process all files in a directory with text splitting."""
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path

    if not dir_path.exists() or not dir_path.is_dir():
        logger.error(f"Directory does not exist: {dir_path}")
        raise NotADirectoryError(f"Directory does not exist: {dir_path}")

    logger.info(f"Starting to process directory with text splitting: {dir_path}")

    # Get supported files
    supported_exts = ['.pdf', '.txt']
    files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if not files:
        logger.warning(f"No supported files found in directory: {dir_path}")
        return []

    results = []
    for file in tqdm(files, desc="Processing and splitting files"):
        try:
            output_file = process_file_with_splitting(file, chunk_size, chunk_overlap)
            results.append(output_file)
        except Exception as e:
            logger.error(f"Failed to process and split file {file}: {e}")

    logger.success(f"Directory processing with splitting completed, processed {len(results)}/{len(files)} files")
    return results


if __name__ == "__main__":
    # Process a single file with text splitting
    process_file_with_splitting(DATA_DIR / "recreational_fishing_guide.pdf", chunk_size, chunk_overlap)
