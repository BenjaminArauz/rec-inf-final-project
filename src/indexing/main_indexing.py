import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CORPUS_DATA_DIR, TFIDF_JSON_PATH, CRAWLER_START_URL, USE_STEMMING
from indexing.crawler import WebCrawler
from indexer import DocumentIndexer


def main():
    """
    Main indexing pipeline: process documents and compute TF-IDF matrix.
    """
    print("=" * 60)
    print("STARTING DOCUMENT INDEXING")
    print("=" * 60)
    
    # Crawl and sync corpus from remote index
    print("Syncing corpus from:", CRAWLER_START_URL)
    crawler = WebCrawler()
    crawler.crawl(CRAWLER_START_URL)
        
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
