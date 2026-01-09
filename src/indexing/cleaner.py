import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
import math

# Import constants from our new config loader
from config import MIN_WORD_LENGTH, LANGUAGE

PREPOSITIONS = {
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
    'about', 'as', 'into', 'like', 'through', 'after',
    'over', 'between', 'out', 'against', 'during', 'without',
    'before', 'under', 'around', 'among', 'within', 'along', 'or'
}


def get_nltk_stopwords():
    """
    Returns the set of stopwords directly from NLTK.
    """
    # We use a set for faster lookup
    return set(stopwords.words(LANGUAGE))


def clean_text(text):
    """
    Clean text: remove regex chars and convert to lowercase.
    
    Parameters:
    - text: str to clean
    
    Returns:
    - str: cleaned and lowercased text
    """
    # PUNCTUATION MARKS
    # Remove the punctuation marks
    text = re.sub(r'[\[\]\{\}.,¿?¡#\$\'\"+<>:;%*]', '', text)

    # Remove parentheses
    text = re.sub(r'[()=]', ' ', text)

    # NUMBERS
    text = re.sub(r'\b\d+\b', '', text)

    # SLASH 
    # Remove multiple slashes
    text = re.sub(r'/{2,}', '/', text)
    # Remove slashes at the start or end of words
    text = re.sub(r'(?<!\S)/|/(?!\S)', '', text)

    # HYPHENS
    # Remove multiple hyphens
    text = re.sub(r'-{2,}', '-', text)
    # Remove hyphens at the start or end of words
    text = re.sub(r'(?<!\S)-|-(?!\S)', '', text)

    # Remove any leftover slashes
    text = re.sub(r'(?<!\S)/|/(?!\S)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    pattern = r'\b(\w+(?:-\w+)*)-(' + '|'.join(PREPOSITIONS) + r')\b';

    def separate_if_prepositions(match):
        base_term = match.group(1)
        preposition = match.group(2)

        base_parts = base_term.split('-')
        if base_parts[-1].lower() in PREPOSITIONS:
            return match.group(0)
        else:
            return f"{base_term} {preposition}"
        
    text = re.sub(pattern, separate_if_prepositions, text, flags=re.IGNORECASE)

    return text

def extract_terms_with_positions(text_original, text_cleaned, stopwords_set):
    """
    Extract terms and their character positions from original text.
    Filters stopwords, applies min length, and stems terms.
    
    Parameters:
    - text_original: original text (lowercased but not cleaned)
    - text_cleaned: cleaned and lowercased text
    - stopwords_set: set of stopwords to filter
    
    Returns:
    - tuple: (final_terms, term_positions)
      - final_terms: list of stemmed terms
      - term_positions: dict mapping term -> [sorted unique positions in ORIGINAL text]
    """
    stemmer = SnowballStemmer(LANGUAGE)
    
    # Tokenize cleaned text to get the final tokens
    tokens_cleaned = word_tokenize(text_cleaned, language=LANGUAGE)
    
    final_terms = []
    term_positions = defaultdict(list)
    current_search_pos = 0
    
    # Process ALL tokens from cleaned text
    for cleaned_token in tokens_cleaned:
        # Find this token in the original text
        # Try exact match first
        token_pos = text_original.find(cleaned_token, current_search_pos)
        
        # If we found a position, advance search; otherwise mark as -1
        if token_pos >= 0:
            current_search_pos = token_pos + 1
        else:
            token_pos = -1
        
        # Filter only by stopwords and min length
        if cleaned_token not in stopwords_set and len(cleaned_token) >= MIN_WORD_LENGTH:
            term_to_add = stemmer.stem(cleaned_token)
            final_terms.append(term_to_add)
            term_positions[term_to_add].append(token_pos)
    
    # Convert lists to sorted unique positions
    positions_dict = {term: sorted(set(pos)) for term, pos in term_positions.items()}
    
    return final_terms, positions_dict

# clean_individual_token removed as per request; matching relies solely on exact finds in original text

def preprocess_text(text):
    """
    Preprocess text: clean, tokenize, filter, and stem.
    Always returns terms with their character positions in the ORIGINAL text.
    
    Parameters:
    - text: str to preprocess
    
    Returns:
    - tuple: (final_terms, term_positions)
      - final_terms: list of stemmed terms
      - term_positions: dict mapping term -> [sorted unique positions in original text]
    """
    stopwords_set = get_nltk_stopwords()
    text_original = text.lower()
    text_cleaned = clean_text(text_original)
    return extract_terms_with_positions(text_original, text_cleaned, stopwords_set)


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