import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Import constants from our new config loader
from config import CHARS_TO_REMOVE_REGEX, MIN_WORD_LENGTH, LANGUAGE

def get_nltk_stopwords():
    """
    Returns the set of stopwords directly from NLTK.
    """
    # We use a set for faster lookup
    return set(stopwords.words(LANGUAGE))

def preprocess_text(text, stopwords_set=None):
    """
    Applies filters: Regex chars, Stopwords (NLTK), Size, Stemming.
    """
    # If stopwords are not passed, load them
    if stopwords_set is None:
        stopwords_set = get_nltk_stopwords()

    # 1. Character Filter (Regex from config)
    text_cleaned = re.sub(CHARS_TO_REMOVE_REGEX, ' ', text)
    text_cleaned = text_cleaned.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text_cleaned, language=LANGUAGE)
    
    # 3. Filtering & Stemming
    stemmer = SnowballStemmer(LANGUAGE)
    final_terms = []
    
    for token in tokens:
        # Check stopword and min length
        if token not in stopwords_set and len(token) >= MIN_WORD_LENGTH:
            term_to_add = stemmer.stem(token)  
            final_terms.append(term_to_add)
            
    return final_terms