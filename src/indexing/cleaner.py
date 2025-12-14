import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
import math


# Import constants from our new config loader
from config import MIN_WORD_LENGTH, LANGUAGE

def get_nltk_stopwords():
    """
    Returns the set of stopwords directly from NLTK.
    """
    # We use a set for faster lookup
    return set(stopwords.words(LANGUAGE))


def clean_text(text, stopwords_set):
    """
    Clean text: remove regex chars and convert to lowercase.
    
    Parameters:
    - text: str to clean
    
    Returns:
    - str: cleaned and lowercased text
    """
    # Remove the punctuation marks
    text = re.sub(r'[().,¿?¡=#\$\'\"+]', ' ', text)
    # Remove all the numbers, except those within words
    text = re.sub(r'\b\d+\b', ' ', text)
    # Remove slashes not within words
    text = re.sub(r'(?<!\S)/(?!\S)|(?<!\S)/|/(?!\S)', ' ', text)
    # Extra: collapse leading or trailing multiple slashes around words (e.g., //enhanc -> enhanc, word// -> word)
    text = re.sub(r'(?<![A-Za-z])/+', '', text)
    text = re.sub(r'/+(?![A-Za-z])', '', text)
    # Hyphen rules: remove leading '-word' and trailing 'word-'
    text = re.sub(r'(?<![A-Za-z0-9])-(?=[A-Za-z])', '', text)
    text = re.sub(r'(?<=[A-Za-z])-(?![A-Za-z0-9])', '', text)
    # Remove hyphen in compounds where the right part is an English stopword (e.g., 'afterglow-of' -> 'afterglow of')
    def _hyphen_compound(m):
        left, right = m.group(1), m.group(2)
        return f"{left} {right}" if right.lower() in stopwords_set else m.group(0)
    text = re.sub(r'([A-Za-z]+)-([A-Za-z]+)', _hyphen_compound, text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def extract_terms_with_positions(text_lower, stopwords_set):
    """
    Extract terms and their character positions from cleaned text.
    Filters stopwords, applies min length, and stems terms.
    
    Parameters:
    - text_lower: cleaned and lowercased text
    
    Returns:
    - tuple: (final_terms, term_positions)
      - final_terms: list of stemmed terms
      - term_positions: dict mapping term -> [sorted unique positions]
    """
    stemmer = SnowballStemmer(LANGUAGE)
    tokens = word_tokenize(text_lower, language=LANGUAGE)
    
    final_terms = []
    term_positions = defaultdict(list)
    current_pos = 0
    
    for token in tokens:
        # Find position of this token in cleaned text
        token_pos = text_lower.find(token, current_pos)
        
        # Check stopword and min length
        """
        final_terms.append(token)
        term_positions[token].append(token_pos)
        """
        
        if token not in stopwords_set and len(token) >= MIN_WORD_LENGTH:
            term_to_add = stemmer.stem(token)
            final_terms.append(term_to_add)
            term_positions[term_to_add].append(token_pos)
        
        
        if token_pos >= 0:
            current_pos = token_pos + len(token)
    
    # Convert lists to sorted unique positions
    positions_dict = {term: sorted(set(pos)) for term, pos in term_positions.items()}
    
    return final_terms, positions_dict


def preprocess_text(text):
    """
    Preprocess text: clean, tokenize, filter, and stem.
    Always returns terms with their character positions.
    
    Parameters:
    - text: str to preprocess
    
    Returns:
    - tuple: (final_terms, term_positions)
      - final_terms: list of stemmed terms
      - term_positions: dict mapping term -> [sorted unique positions]
    """
    stopwords_set = get_nltk_stopwords()
    text_lower = clean_text(text, stopwords_set)
    return extract_terms_with_positions(text_lower, stopwords_set)


def compute_tf(text):
    """
    Compute TF (Term Frequency) with character positions using the formula:
    - If frecuencia > 0: TF = 1 + log2(frecuencia)
    - Else: TF = 0
    
    Parameters:
    - text: str (raw text to process)
    
    Returns:
    - dict mapping term -> {'tf': value, 'positions': [pos1, pos2, ...]}
    """
    # Preprocess always returns terms and positions
    terms, positions = preprocess_text(text)
    
    # Count occurrences
    term_counts = Counter(terms)
    
    tf = {}
    for term, frequency in term_counts.items():
        tf_val = 1.0 + math.log2(frequency) if frequency > 0 else 0.0
        tf[term] = {
            'tf': tf_val,
            'positions': positions[term]
        }
    
    return tf