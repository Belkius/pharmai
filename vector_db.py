import os
import re
import time
from pathlib import Path
from typing import List
from pymilvus import MilvusClient, DataType, utility
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker

from pymilvus.model.hybrid import MGTEEmbeddingFunction

# client = MilvusClient(host="localhost", port=1953, timeout=10)
client = MilvusClient("milvus_db.db")

def initialize_milvus_collection(client, collection_name="CHPL_collection"):
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="availability_categories", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=20, max_capacity=5)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="COSINE", index_name="dense_index")
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", index_name="sparse_index")

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print(f"Collection '{collection_name}' created successfully.")

def get_existing_ids(client, collection_name):
    results = client.query(
        collection_name=collection_name,
        filter="id >= 0",
        output_fields=["id"]
    )
    existing_ids = {res["id"] for res in results}
    return existing_ids

def extract_categories_and_text(file_path, df_cat):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    id_match = re.search(r'char_(\d+)', file_path.name)
    if not id_match:
        raise ValueError(f"ID not found in file name: {file_path.name}")

    id_value = int(id_match.group(1))

    raw_categories = df_cat.loc[df_cat["Identyfikator Produktu Leczniczego"] == id_value]["Kategorie dostępności"]
    if raw_categories.empty:
        categories = []
    else:
        raw_categories = raw_categories.values[0]
        if isinstance(raw_categories, str):
            categories = [cat.strip() for cat in raw_categories.split(",")]
        else:
            categories = []

    return categories, content, id_value

mgte_ef = MGTEEmbeddingFunction(
    model_name="Alibaba-NLP/gte-multilingual-base",
)

def prepare_data_for_insertion(file_paths, df_cat, existing_ids):
    all_texts = []
    all_categories = []
    all_ids = []

    for file_path in file_paths:
        categories, text, id_value = extract_categories_and_text(file_path, df_cat)
        if id_value in existing_ids:
            continue
        all_texts.append(text)
        all_categories.append(categories)
        all_ids.append(id_value)

    if not all_texts:
        print("No new data to insert.")
        return []

    vectors = mgte_ef.encode_documents(all_texts)

    data = []
    for idx, text in enumerate(all_texts):
        indices = vectors['sparse'][idx].col.tolist()
        values = vectors['sparse'][idx].data.tolist()
        sparse_vector_data = list(zip(indices, values))

        entry = {
            "id": all_ids[idx],
            "text": text,
            "dense_vector": vectors['dense'][idx].tolist(),
            "sparse_vector": sparse_vector_data,
            "availability_categories": all_categories[idx]
        }
        data.append(entry)

    return data

def insert_data(client, collection_name, data):
    if data:
        client.insert(collection_name=collection_name, data=data)
        print("Data inserted successfully into Milvus.")
    else:
        print("No data to insert.")

def process_in_batches(client, file_paths, df_cat, collection_name="CHPL_collection", batch_size=40, pause_time=20):
    existing_ids = get_existing_ids(client, collection_name)
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}: {len(batch_files)} files")

        data = prepare_data_for_insertion(batch_files, df_cat, existing_ids)
        if data:
            insert_data(client, collection_name, data)
            existing_ids.update(entry["id"] for entry in data)
        else:
            print("No new data to insert in this batch.")
            continue

        print(f"Waiting {pause_time} seconds before next batch...")
        time.sleep(pause_time)

def query_hybrid_search(query: str, limit: int = 2, collection_name="CHPL_collection"):
    query_embeddings = mgte_ef([query])

    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"], "sparse_vector", param={"metric_type": "IP"}, limit=10
    )
    dense_req = AnnSearchRequest(
        query_embeddings["dense"], "dense_vector", {"metric_type": "COSINE"}, limit=10
    )

    result = client.hybrid_search(
        collection_name=collection_name, reqs=[sparse_req, dense_req], ranker=RRFRanker(), limit=limit, output_fields=["text"]
    )

    return result

def query_dense_search(query: str, limit: int = 2, collection_name="CHPL_collection"):
    query_embeddings = mgte_ef([query])
    dense_vec = query_embeddings['dense'][0].tolist()

    search_params = {
        "collection_name": collection_name,
        "data": [dense_vec],
        "anns_field": "dense_vector",
        "params":{},
        "metric_type": "COSINE",
        "limit": limit,
        "output_fields": ["id"],
    }
    res_dense = client.search(**search_params)[0]

    return res_dense

def extract_ids(data):
    """
    Extracts 'id' values from a nested list of result dictionaries.

    Args:
        data (list): A nested list in the form [[{...}, {...}, ...]].

    Returns:
        list: A flat list of IDs.
    """
    return [item['id'] for sublist in data for item in sublist]

import os

def load_texts_from_ids(ids, directory="charakterystyki_md"):
    """
    Loads text content from Markdown files named 'char_<ID>.md'.

    Args:
        ids (list): List of integer IDs.
        directory (str): Directory where the markdown files are located.

    Returns:
        dict: A dictionary mapping ID to the content of its Markdown file.
    """
    content_dict = {}
    for id_ in ids:
        filename = os.path.join(directory, f"char_{id_}.md")
        try:
            with open(filename, "r", encoding="utf-8") as file:
                content_dict[id_] = file.read()
        except FileNotFoundError:
            content_dict[id_] = None
    return content_dict

from typing import List

import re
from typing import List

def headers_extraction(document_text: str, relevant_headers: List[str]) -> List[str]:
    """
    Extract sections of the document that match relevant headers,
    using the known order of all headers to find section boundaries.

    Args:
        document_text (str): Full document text to parse.
        relevant_headers (List[str]): List of relevant headers to match.

    Returns:
        List[str]: List of matched document sections.
    """
    ordered_headers = [
        "NAZWA PRODUKTU LECZNICZEGO",
        "SKŁAD JAKOŚCIOWY I ILOŚCIOWY",
        "POSTAĆ FARMACEUTYCZNA",
        "SZCZEGÓŁOWE DANE KLINICZNE",
        "WŁAŚCIWOŚCI FARMAKOLOGICZNE",
        "DANE FARMACEUTYCZNE",
        "PODMIOT ODPOWIEDZIALNY POSIADAJĄCY POZWOLENIE NA DOPUSZCZENIE DO OBROTU",
        "NUMER POZWOLENIA NA DOPUSZCZENIE DO OBROTU",
        "DATA WYDANIA PIERWSZEGO POZWOLENIA NA DOPUSZCZENIE DO OBROTU I DATA PRZEDŁUŻENIA POZWOLENIA",
        "DATA ZATWIERDZENIA LUB CZĘŚCIOWEJ ZMIANY TEKSTU CHARAKTERYSTYKI PRODUKTU LECZNICZEGO"
    ]

    relevant_headers_lower = set(h.lower() for h in relevant_headers)
    name_header_lower = ordered_headers[0].lower()

    relevant_headers_lower.add(name_header_lower)

    header_pattern = re.compile(r'^#+\s*(\d+\.)?\s*(.+)', re.IGNORECASE)

    lines = document_text.splitlines()
    sections = []
    current_section = None

    header_indices = []
    for i, line in enumerate(lines):
        match = header_pattern.match(line.strip())
        if match:
            header_text = match.group(2).strip().lower()
            for ordered_header in ordered_headers:
                if ordered_header.lower() in header_text:
                    header_indices.append((i, ordered_header))
                    break

    for idx, (start_idx, header_text) in enumerate(header_indices):
        end_idx = header_indices[idx + 1][0] if idx + 1 < len(header_indices) else len(lines)
        if header_text.lower() in relevant_headers_lower:
            content = "\n".join(lines[start_idx:end_idx]).strip()
            sections.append(content)

    return sections


def check_collection(client=client, collection_name="CHPL_collection"):
    """
    Check if the collection exists in the vector database.
    """

    if client.has_collection(collection_name=collection_name):
        print("Collection exists")
    else:
        print("no collection")