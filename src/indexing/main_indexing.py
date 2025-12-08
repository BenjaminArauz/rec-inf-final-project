import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CORPUS_DATA_DIR, TFIDF_JSON_PATH
from indexer import DocumentIndexer


def main():
    """
    Main indexing pipeline: process documents and compute TF-IDF matrix.
    """
    print("=" * 60)
    print("STARTING DOCUMENT INDEXING")
    print("=" * 60)
    
    # Initialize indexer and build complete index
    indexer = DocumentIndexer(CORPUS_DATA_DIR)
    tfidf_matrix = indexer.build_index()

    # Save JSON in processed folder
    indexer.save_json(TFIDF_JSON_PATH)
    print(f"TF-IDF JSON saved to: {TFIDF_JSON_PATH}")
    
    # Print summary
    #indexer.print_summary()
    
    return tfidf_matrix


if __name__ == "__main__":
    tfidf = main()
