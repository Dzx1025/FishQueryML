import os
import json
import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus
from FishQueryML.utils.constants import DATA_DIR
from typing import Dict, List, Iterable
from pathlib import Path

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def process_pdfs_with_docling(pdf_dir: Path) -> Dict[str, str]:
    """
    Process all PDF files in a directory using docling batch conversion.

    Args:
        pdf_dir (Path): Directory containing PDF files

    Returns:
        Dict[str, str]: Dictionary mapping page number to document content
    """
    # Collect all PDF paths
    pdf_paths = []
    page_numbers = {}  # Store mapping of file path to page number

    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            # Extract page number from filename (e.g., "original_page_1.pdf" -> "1")
            page_num = filename.split('_')[-1].replace('.pdf', '')
            file_path = pdf_dir / filename
            pdf_paths.append(file_path)
            page_numbers[str(file_path)] = page_num

    results = {}
    if not pdf_paths:
        _log.warning(f"No PDF files found in {pdf_dir}")
        return results

    # Initialize converter and process all documents
    doc_converter = DocumentConverter()
    conv_results = doc_converter.convert_all(
        pdf_paths,
        raises_on_error=False
    )

    # Process conversion results
    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        page_num = page_numbers[str(conv_res.input.file)]

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            results[page_num] = conv_res.document.export_to_markdown()

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(f"Document page {page_num} was partially converted with the following errors:")
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            results[page_num] = conv_res.document.export_to_markdown()
            partial_success_count += 1

        else:
            _log.info(f"Document page {page_num} failed to convert.")
            results[page_num] = f"Error: Conversion failed"
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} pages, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )

    return results


def save_results_to_json(results: Dict[str, str], output_path: Path):
    """
    Save the processing results to a JSON file.

    Args:
        results (Dict[str, str]): Dictionary of processing results
        output_path (Path): Path to save the JSON file
    """
    try:
        # Sort results by page number
        sorted_results = dict(sorted(results.items(), key=lambda x: int(x[0])))

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        _log.info(f"Results saved to: {output_path}")
    except Exception as e:
        _log.error(f"Error saving results: {str(e)}")


def main():
    # Directory containing split PDFs
    pdf_dir = DATA_DIR / 'tmp'

    # Output JSON file path
    output_path = DATA_DIR / 'output' / 'results.json'

    # Process PDFs
    if pdf_dir.exists():
        results = process_pdfs_with_docling(pdf_dir)

        # Save results
        save_results_to_json(results, output_path)
    else:
        _log.error(f"PDF directory not found: {pdf_dir}")


if __name__ == "__main__":
    main()
