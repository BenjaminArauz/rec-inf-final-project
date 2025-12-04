import sys
import os

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CORPUS_DATA_DIR
from cleaner import preprocess_text, get_nltk_stopwords

def main():
    print("--- Starting Preprocessing (NLTK + Config File) ---\n")

    # 1. Load NLTK stopwords once (for efficiency)
    stop_words = get_nltk_stopwords()
    print(f"Loaded {len(stop_words)} stopwords from NLTK.")

    # 2. Process Files
    if not os.path.exists(CORPUS_DATA_DIR):
        print(f"Error: Folder {CORPUS_DATA_DIR} not found.")
        return

    files = os.listdir(CORPUS_DATA_DIR)
    
    for filename in files:
        filepath = os.path.join(CORPUS_DATA_DIR, filename)
        
        if os.path.isfile(filepath):
            print(f"\nProcessing file: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clean the text
            terms = preprocess_text(content, stopwords_set=stop_words)
            
            print(f"Terms found: {terms}")
            break

if __name__ == "__main__":
    main()