import os
from PyPDF2 import PdfReader, PdfWriter
from FishQueryML.utils.constants import DATA_DIR


def split_pdf(input_pdf_path):
    """
    Split a PDF file into individual pages and save them in DATA_DIR/tmp directory.

    Args:
        input_pdf_path (str): Path to the input PDF file

    Returns:
        str: Path to the directory containing split PDF files
    """
    # Create tmp directory if it doesn't exist
    output_dir = DATA_DIR / 'tmp'
    output_dir.mkdir(exist_ok=True)

    try:
        # Open the PDF file
        pdf_reader = PdfReader(input_pdf_path)
        total_pages = len(pdf_reader.pages)

        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(input_pdf_path))[0]

        # Split each page into a separate PDF
        for page_num in range(total_pages):
            # Create a PDF writer object
            pdf_writer = PdfWriter()

            # Add the current page
            pdf_writer.add_page(pdf_reader.pages[page_num])

            # Generate output filename
            output_filename = f"{base_filename}_page_{page_num + 1}.pdf"
            output_path = output_dir / output_filename

            # Write the split PDF to file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)

        print(f"PDF has been split into {total_pages} files")
        print(f"Files are saved in: {output_dir}")
        return output_dir

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_path = DATA_DIR / "origin.pdf"
    output_dir = split_pdf(pdf_path)

    if output_dir:
        print("\nSplit PDF files:")
        for filename in os.listdir(output_dir):
            print(f"- {filename}")
