import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import TFIDF_JSON_PATH
from search_engine import SearchEngine


def display_menu():
    """
    Display search type menu and get user selection.
    
    Returns:
    - str: 'AND' or 'OR'
    """
    print("\n" + "=" * 60)
    print("SELECT SEARCH TYPE")
    print("=" * 60)
    print("1. AND Search - Find documents containing ALL terms")
    print("2. OR Search  - Find documents containing AT LEAST ONE term")
    print("=" * 60)
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            return 'AND'
        elif choice == '2':
            return 'OR'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_search_terms():
    """
    Prompt user for search terms.
    
    Returns:
    - str: user's search terms
    """
    terms = input("\nEnter search terms (space-separated): ").strip()
    return terms


def build_query(search_type, terms):
    """
    Build query string (just returns the terms, operator is passed separately).
    
    Parameters:
    - search_type: 'AND' or 'OR' (not used anymore, kept for compatibility)
    - terms: str space-separated terms
    
    Returns:
    - str: terms as-is
    """
    return terms.strip()


def main():
    """
    Main searching pipeline: load index and process user queries.
    """
    print("=" * 60)
    print("DOCUMENT SEARCH SYSTEM")
    print("=" * 60)
    
    # Initialize search engine
    engine = SearchEngine(TFIDF_JSON_PATH)
    
    # Load TF-IDF index
    print(f"\nLoading TF-IDF index from: {TFIDF_JSON_PATH}")
    engine.load_index()
    
    # Main search loop
    while True:
        # Display menu and get search type
        search_type = display_menu()
        print(f"\nSelected: {search_type} search")
        
        # Get search terms
        terms = get_search_terms()
        
        if not terms:
            print("No search terms provided.")
            retry = input("\nTry another search? (y/n): ").strip().lower()
            if retry != 'y':
                break
            continue
        
        # Build query
        query = build_query(search_type, terms)
        print(f"\nSearch terms: '{query}'")
        
        # Search with operator
        results = engine.search(query, operator=search_type)
        
        # Ask if user wants to search again
        print("\n" + "-" * 60)
        retry = input("\nTry another search? (y/n): ").strip().lower()
        if retry != 'y':
            break
    
    print("\nThank you for using the Document Search System!")
    print("=" * 60)


if __name__ == "__main__":
    main()
