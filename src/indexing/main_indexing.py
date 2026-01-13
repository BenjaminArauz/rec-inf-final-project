import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CORPUS_DATA_DIR, TFIDF_JSON_PATH, REMOTE_INDEX_URL, USE_STEMMING
from indexing.corpus_crawler import sync_corpus_from_index
from indexer import DocumentIndexer


def main():
    """
    Main indexing pipeline: process documents and compute TF-IDF matrix.
    """
    print("=" * 60)
    print("STARTING DOCUMENT INDEXING")
    print("=" * 60)
    
    # Crawl and sync corpus from remote index
    print("Syncing corpus from:", REMOTE_INDEX_URL)
    summary = sync_corpus_from_index(REMOTE_INDEX_URL, CORPUS_DATA_DIR)
    print(f"Corpus sync: total={summary['total']} downloaded={summary['downloaded']} skipped={summary['skipped']}")

    # Show stemming status
    if USE_STEMMING:
        print("Stemming is ENABLED for indexing.")
    else:
        print("Stemming is DISABLED for indexing.")

    # Initialize indexer and build complete index
    indexer = DocumentIndexer(CORPUS_DATA_DIR)
    tfidf_matrix = indexer.build_index()

    # Save JSON in processed folder
    indexer.save_json(TFIDF_JSON_PATH)
    print(f"TF-IDF JSON saved to: {TFIDF_JSON_PATH}")
    
    return tfidf_matrix


if __name__ == "__main__":
    tfidf = main()
