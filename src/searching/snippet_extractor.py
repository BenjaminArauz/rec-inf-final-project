"""
Text snippet extraction module for displaying search results.
"""
import os


def extract_snippet(text, term_position, max_words=15):
    """
    Extract a text fragment around a specific character position.
    Priority:
    1. Extract complete sentence (from period to period)
    2. If no periods or exceeds limit, use other punctuation marks (comma, semicolon, etc.)
    3. Otherwise, cut at max_words limit
    
    Parameters:
    - text: str full document text
    - term_position: int character position where term starts
    - max_words: int maximum number of words to include (default: 15)
    
    Returns:
    - tuple: (fragment, term_start_in_fragment, term_length) or None
    """
    if term_position >= len(text):
        return None
    
    # Find the term at this position (until next space or end)
    term_end = term_position
    while term_end < len(text) and not text[term_end].isspace():
        term_end += 1
    
    term_length = term_end - term_position
    
    # Priority 1: Try to find sentence boundaries (period to period)
    sentence_start = term_position
    sentence_end = term_end
    
    # Find start of sentence (look back for period)
    found_start_period = False
    while sentence_start > 0:
        if text[sentence_start - 1] in '.!?':
            found_start_period = True
            break
        sentence_start -= 1
    
    # Skip whitespace at start
    while sentence_start < len(text) and text[sentence_start].isspace():
        sentence_start += 1
    
    # Find end of sentence (look forward for period)
    found_end_period = False
    while sentence_end < len(text):
        if text[sentence_end] in '.!?':
            sentence_end += 1  # Include the period
            found_end_period = True
            break
        sentence_end += 1
    
    # If we found both periods, check word count
    if found_start_period or found_end_period or sentence_start == 0:
        full_sentence = text[sentence_start:sentence_end].strip()
        word_count = len(full_sentence.split())
        
        if word_count <= max_words:
            # Priority 1: Return full sentence if it fits
            term_start_in_fragment = term_position - sentence_start
            return (full_sentence, term_start_in_fragment, term_length)
    
    # Priority 2: No periods or sentence too long, use other punctuation
    start = sentence_start
    end = sentence_end
    
    # Look for punctuation before term
    for i in range(sentence_start, term_position):
        if text[i] in ',;:':
            temp_start = i + 1
            while temp_start < len(text) and text[temp_start].isspace():
                temp_start += 1
            
            temp_fragment = text[temp_start:end].strip()
            if len(temp_fragment.split()) <= max_words:
                start = temp_start
    
    # Look for punctuation after term
    for i in range(term_end, sentence_end):
        if text[i] in ',;:':
            temp_fragment = text[start:i+1].strip()
            if len(temp_fragment.split()) <= max_words:
                end = i + 1
                break
    
    fragment = text[start:end].strip()
    
    # Priority 3: Still too long, hard cut at max_words
    if len(fragment.split()) > max_words:
        words = fragment.split()
        fragment = ' '.join(words[:max_words])
    
    term_start_in_fragment = term_position - start
    return (fragment, term_start_in_fragment, term_length)


def highlight_term_in_fragment(fragment, term_start, term_length):
    """
    Highlight the search term in the fragment using exact position.
    
    Parameters:
    - fragment: str text fragment
    - term_start: int position where term starts in fragment
    - term_length: int length of the term
    
    Returns:
    - str: fragment with highlighted term
    """
    if not fragment or term_start < 0 or term_start >= len(fragment):
        return fragment
    
    # Extract the actual term from the fragment
    term_end = min(term_start + term_length, len(fragment))
    term = fragment[term_start:term_end]
    
    # Build highlighted fragment
    before = fragment[:term_start]
    highlighted_term = f"**{term.upper()}**"
    after = fragment[term_end:]
    
    return before + highlighted_term + after


def get_snippet_for_term(corpus_dir, doc_id, term_positions, max_words=15):
    """
    Get a highlighted text snippet for the first occurrence of a term.
    
    Parameters:
    - corpus_dir: str path to corpus directory
    - doc_id: str document identifier
    - term_positions: list of character positions
    - max_words: int maximum number of words (default: 15)
    
    Returns:
    - str: highlighted snippet or None
    """
    if not term_positions:
        return None
    
    filepath = os.path.join(corpus_dir, doc_id)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use first occurrence
        first_position = term_positions[0]
        
        # Extract snippet with term position info
        result = extract_snippet(text, first_position, max_words)
        
        if result:
            fragment, term_start, term_length = result
            return highlight_term_in_fragment(fragment, term_start, term_length)
        
        return None
    
    except Exception as e:
        return None
