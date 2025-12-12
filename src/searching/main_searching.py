import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import TFIDF_JSON_PATH, CORPUS_DATA_DIR
from search_engine import SearchEngine
    

def display_menu():
    """
    Display search type menu and get user selection.
    
    Returns:
    - str: 'AND', 'OR', or 'PHRASE'
    """
    print("SELECT SEARCH TYPE")
    print("=" * 60)
    print("1. AND Search    - Find documents containing ALL terms")
    print("2. OR Search     - Find documents containing AT LEAST ONE term")
    print("3. PHRASE Search - Find documents containing the exact phrase")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            return 'AND'
        elif choice == '2':
            return 'OR'
        elif choice == '3':
            return 'PHRASE'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def display_results(results):
    """
    Display search results with snippets.
    
    Parameters:
    - results: list of result dictionaries with 'doc', 'score', and 'snippets'
    """
    if not results:
        print("\n  No results found.")
        return
    
    print(f"\n{'='*60}")
    print(f"SEARCH RESULTS ({len(results)} document(s) found)")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Document ID: {result['doc']} (weight: {result['score']:.4f})")
        
        # Display snippets for each term
        if result.get('snippets'):
            print(f"Text Fragments:")
            for term, snippet in result['snippets'].items():
                print(snippet)
        else:
            print(f"No text fragments available.")
        
        print()


def main():
    """
    Main searching pipeline: load index and process user queries.
    """
    print("=" * 60)
    print("DOCUMENT SEARCH SYSTEM")
    print("=" * 60)
    
    # Initialize search engine
    engine = SearchEngine(TFIDF_JSON_PATH, CORPUS_DATA_DIR)
    
    # Load TF-IDF index
    print(f"\nLoading TF-IDF index from: {TFIDF_JSON_PATH}")
    engine.load_index()
    
    # Main search loop
    while True:
        # Display menu and get search type
        search_type = display_menu()
        print(f"\nSelected: {search_type} search")
        
        # Get search terms
        terms = input("\nEnter search terms (space-separated): ").strip()
        
        if not terms:
            print("No search terms provided.")
            retry = input("\nTry another search? (y/n): ").strip().lower()
            if retry != 'y':
                break
            continue
        
        # Build query
        query = terms
        print(f"\nSearch terms: '{query}'")
        
        # Search with operator
        results = engine.search(query, operator=search_type)
        
        # Display results with snippets
        display_results(results)
        
        # Ask if user wants to search again
        print("\n" + "-" * 60)
        retry = input("\nTry another search? (y/n): ").strip().lower()
        if retry != 'y':
            break
    
    print("\nThank you for using the Document Search System!")


if __name__ == "__main__":
    main()
