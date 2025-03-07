import os
from sentence_transformers import SentenceTransformer, util
import sys

def load_text(file_path: str) -> str:
    """
    Loads text from a file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_clauses(text: str) -> list:
    """
    Splits a document's text into clauses. Here, we assume that clauses
    are separated by two newline characters.
    """
    clauses = [clause.strip() for clause in text.split("\n\n") if clause.strip()]
    return clauses

def encode_text(model, text, is_multiple_texts=False):
    """
    Encodes text using the provided model.
    """
    return model.encode(text, convert_to_tensor=True)

def calculate_similarity(sop_embedding, clause_embeddings):
    """
    Calculates cosine similarity between SOP embedding and clause embeddings.
    """
    return util.cos_sim(sop_embedding, clause_embeddings)[0]

def find_relevant_clauses(sop_text: str, clauses: list, model, threshold=0.4) -> list:
    """
    Compares each clause from the regulatory document to the entire SOP text.
    Returns a list of tuples containing the clause and its similarity score if
    the score is above the threshold.
    """
    relevant = []
    sop_embedding = encode_text(model, sop_text)
    clause_embeddings = encode_text(model, clauses, is_multiple_texts=True)
    
    # Compute cosine similarity between the SOP and each clause
    cosine_scores = calculate_similarity(sop_embedding, clause_embeddings)
    
    for clause, score in zip(clauses, cosine_scores):
        if score >= threshold:
            relevant.append((clause, float(score)))
    return relevant

def process_single_regulatory_doc(sop_text, reg_file_path, model):
    """
    Processes a single regulatory document and finds clauses relevant to the SOP.
    """
    regulatory_text = load_text(reg_file_path)
    clauses = extract_clauses(regulatory_text)
    return find_relevant_clauses(sop_text, clauses, model)

def process_regulatory_docs(sop_file: str, regulatory_docs_folder: str):
    """
    Processes all regulatory documents in the specified folder. For each document,
    it extracts clauses and finds those relevant to the SOP.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sop_text = load_text(sop_file)
    
    all_relevant_clauses = []
    
    for filename in os.listdir(regulatory_docs_folder):
        if filename.lower().endswith("_extracted.txt"):
            file_path = os.path.join(regulatory_docs_folder, filename)
            relevant_clauses = process_single_regulatory_doc(sop_text, file_path, model)
            if relevant_clauses:
                all_relevant_clauses.append((filename, relevant_clauses))
    return all_relevant_clauses

def print_relevant_clauses(doc_name, clauses):
    """
    Prints relevant clauses with their scores.
    """
    print(f"\nRelevant clauses from {doc_name}:")
    for clause, score in clauses:
        print(f"Score: {score:.3f}\nClause: {clause}\n{'-'*40}")

def check_paths(sop_file, regulatory_folder):
    """
    Checks if the paths to the SOP file and regulatory folder exist.
    """
    if not os.path.exists(sop_file):
        print(f"SOP file '{sop_file}' not found!")
        return False
    if not os.path.isdir(regulatory_folder):
        print(f"Regulatory documents folder '{regulatory_folder}' not found!")
        return False
    return True

def process_specific_reg_file(sop_file, reg_file_path):
    """
    Processes a specific regulatory file.
    """
    if not os.path.exists(reg_file_path):
        print(f"Regulatory file '{reg_file_path}' not found!")
        return None
        
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sop_text = load_text(sop_file)
    return process_single_regulatory_doc(sop_text, reg_file_path, model)