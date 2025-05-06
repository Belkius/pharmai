import os
from typing import List
from pymilvus import MilvusClient
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker

from pymilvus.model.hybrid import MGTEEmbeddingFunction

# client = MilvusClient(host="localhost", port=1953, timeout=10)
client = MilvusClient("milvus_db.db")

mgte_ef = MGTEEmbeddingFunction(
    model_name="Alibaba-NLP/gte-multilingual-base",
)

def query_hybrid_search(query: str, limit: int = 2):
    query_embeddings = mgte_ef([query])

    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"], "sparse_vector", param={"metric_type": "IP"}, limit=10
    )
    dense_req = AnnSearchRequest(
        query_embeddings["dense"], "dense_vector", {"metric_type": "COSINE"}, limit=10
    )

    result = client.hybrid_search(
        collection_name="hybrid_collection", reqs=[sparse_req, dense_req], ranker=RRFRanker(), limit=limit, output_fields=["text"]
    )

    return result

def query_dense_search(query: str, limit: int = 2):
    query_embeddings = mgte_ef([query])
    dense_vec = query_embeddings['dense'][0].tolist()

    search_params = {
        "collection_name": "hybrid_collection",
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
            content_dict[id_] = None  # or handle as appropriate
    return content_dict

def headers_extraction(document_text: str, relevant_headers: List[str]) -> List[str]:
    """
    Extract sections of the document that match the relevant headers.

    Args:
        document_text (str): Full document text to parse.
        relevant_headers (List[str]): List of header strings to match.

    Returns:
        List[str]: Filtered document sections corresponding to relevant headers.
    """
    MUST_INCLUDE_HEADER = "NAZWA PRODUKTU LECZNICZEGO"
    headers_to_check = set(h.lower() for h in relevant_headers)
    headers_to_check.add(MUST_INCLUDE_HEADER.lower())

    doc_sections = document_text.split('\n')
    filtered_sections = []
    current_header = None
    current_content = []

    for line in doc_sections:
        line_stripped = line.strip()
        if any(h.lower() in line_stripped.lower() for h in relevant_headers):
            # Save previous matching section
            if current_header and current_header in relevant_headers and current_content:
                filtered_sections.append(f"{current_header}\n" + "\n".join(current_content))
            current_header = line_stripped
            current_content = []
        elif current_header:
            current_content.append(line_stripped)

    if current_header and current_header in relevant_headers and current_content:
        filtered_sections.append(f"{current_header}\n" + "\n".join(current_content))

    return filtered_sections

def check_collection():
    """
    Check if the collection exists in the vector database.
    """
    from pymilvus import MilvusClient

    client = MilvusClient("milvus_db.db")

    if client.has_collection(collection_name="hybrid_collection"):
        print("Collection exists")
    else:
        print("no collection")