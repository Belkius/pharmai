import os
import re
import time
import requests
from typing import Union

import pandas as pd
from tqdm import tqdm
from docling.document_converter import DocumentConverter


def download_csv(url: str, filename: str) -> None:
    """
    Downloads a CSV file from the given URL and saves it to the specified filename.

    Args:
        url (str): The URL to download the CSV from.
        filename (str): The name of the file to save the CSV content to.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        file.write(response.content)


def extract_categories(opakowanie: Union[str, float]) -> str:
    """
    Extracts availability categories from the 'opakowanie' text.

    Args:
        opakowanie (Union[str, float]): The text containing category information.
                                        Can be NaN (float) or a string.

    Returns:
        str: A comma-separated string of found categories, or an empty string if none found.
    """
    if pd.isna(opakowanie):
        return ""

    # Define the availability categories to search for
    categories = ["OTC", "Rp", "Rpz", "Rpw", "Lz"]
    pattern = r'\b(?:' + '|'.join(map(re.escape, categories)) + r')\b'

    matches = re.findall(pattern, opakowanie)
    if matches:
        return ", ".join(sorted(set(matches)))
    return ""


def process_and_save_categories(
    df: pd.DataFrame,
    opakowanie_col: str = "Opakowanie",
    id_col: str = "Identyfikator Produktu Leczniczego",
    output_filename: str = "Rejestr_Produktow_Leczniczych_z_kategoriami.csv"
) -> None:
    """
    Processes the DataFrame to extract availability categories and saves the result to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        opakowanie_col (str): The name of the column containing packaging information.
        id_col (str): The name of the identifier column.
        output_filename (str): The name of the output CSV file.
    """
    # Create new column with extracted categories
    df["Kategorie dostępności"] = df[opakowanie_col].apply(extract_categories)

    # Save to CSV
    df[[id_col, "Kategorie dostępności"]].to_csv(
        output_filename, sep=';', quotechar='"', index=False
    )


def extract_human_medicines_with_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts human medicines that have availability categories and non-empty characteristics.

    Args:
        df (pd.DataFrame): The input DataFrame containing medicinal product data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with 'Product Identifier' and 'Characteristics' columns.
    """
    # Filter rows where 'Type of preparation' is 'Human'
    df_human = df[df['Rodzaj preparatu'] == 'Ludzki']

    # Remove rows without availability categories
    missing_category_indices = [index for index, row in df_human.iterrows() if not row["Kategorie dostępności"]]
    df_filtered = df_human.drop(missing_category_indices)

    # Select only the necessary columns
    df_filtered = df_filtered[['Identyfikator Produktu Leczniczego', 'Charakterystyka']]

    # Remove rows with empty 'Characteristics'
    df_filtered = df_filtered.dropna(subset=['Charakterystyka'])

    return df_filtered


def download_pdf(url: str, timeout: int = 30) -> bytes | None:
    """
    Downloads the content of a PDF file from a given URL.

    Args:
        url (str): The URL to download the PDF from.
        timeout (int): Timeout in seconds for the request.

    Returns:
        bytes | None: The PDF content if successful, otherwise None.
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None

def save_pdf(pdf_content: bytes, filename: str) -> None:
    """
    Saves the PDF content to a file.

    Args:
        pdf_content (bytes): The PDF content to save.
        filename (str): The output filename.
    """
    with open(filename, 'wb') as f:
        f.write(pdf_content)

def download_pdfs_from_dataframe(df: pd.DataFrame,
                                  url_column: str = 'Charakterystyka',
                                  id_column: str = 'Identyfikator Produktu Leczniczego',
                                  output_dir: str = 'charakterystyki_pdf',
                                  delay: int = 2) -> None:
    """
    Downloads PDFs from links stored in a DataFrame and saves them to a directory.

    Args:
        df (pd.DataFrame): The DataFrame containing URLs and identifiers.
        url_column (str): The name of the column containing the PDF URLs.
        id_column (str): The name of the column containing product identifiers.
        output_dir (str): Directory to save the downloaded PDFs.
        delay (int): Delay in seconds between downloads to avoid API rate limits.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pdf_links = df[url_column]
    print("Starting PDF downloads...")

    for i, link in enumerate(tqdm(pdf_links)):
        try:
            id_number = df.iloc[i][id_column]
            filename = os.path.join(output_dir, f"char_{id_number}.pdf")

            # Skip if already downloaded
            if os.path.exists(filename):
                continue

            pdf_content = download_pdf(link)
            if pdf_content is None:
                continue

            save_pdf(pdf_content, filename)

            # Delay between requests
            time.sleep(delay)

        except Exception as e:
            print(f"Error downloading file {i+1}: {str(e)}")

    print("\nFinished downloading PDF files.")
    print(f"Files saved in directory '{output_dir}'.")

def convert_pdfs_to_markdown(pdf_dir: str, md_dir: str) -> None:
    """
    Converts all PDF files from the input directory to Markdown files.

    Args:
        pdf_dir (str): Directory containing input PDF files.
        md_dir (str): Directory where Markdown files will be saved.
    """
    # Ensure that the output directory exists
    if not os.path.exists(md_dir):
        os.makedirs(md_dir)

    print(f"Starting PDF to Markdown conversion from '{pdf_dir}' to '{md_dir}'...")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            md_filename = pdf_file.replace('.pdf', '.md')
            md_path = os.path.join(md_dir, md_filename)

            if os.path.exists(md_path):
                continue  # Skip if already converted

            converter = DocumentConverter()
            doc = converter.convert(pdf_path)
            md_content = doc.document.export_to_markdown()

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        except Exception as e:
            print(f"Error converting {pdf_file}: {str(e)}")

    print("\nFinished converting PDFs to Markdown.")
    print(f"Markdown files saved in '{md_dir}'.")
