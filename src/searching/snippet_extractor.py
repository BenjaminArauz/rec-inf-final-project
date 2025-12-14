"""
Text snippet extraction module for displaying search results.
"""
import os


def extract_snippet(text, term_position, fragment_size):
    """
    Extract a text fragment around a specific character position.
    
    Parameters:
    - text: str full document text
    - term_position: int character position where term starts
    - fragment_size: int number of words to include in fragment
    
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
    
    # Go back to find start of fragment (target: half of fragment_size words before)
    start = term_position
    words_before = 0
    target_words_before = fragment_size // 2
    
    while start > 0 and words_before < target_words_before:
        start -= 1
        if text[start].isspace() and start > 0 and not text[start - 1].isspace():
            words_before += 1
    
    # Adjust to start of word
    while start > 0 and not text[start].isspace():
        start -= 1
    if start > 0 and text[start].isspace():
        start += 1
    
    # Go forward to find end of fragment
    end = term_end
    words_after = 0
    target_words_after = fragment_size - target_words_before
    
    while end < len(text) and words_after < target_words_after:
        if text[end].isspace() and end > 0 and not text[end - 1].isspace():
            words_after += 1
        end += 1
    
    # Adjust to end of word
    while end < len(text) and not text[end].isspace():
        end += 1
    
    # Extract fragment
    fragment = text[start:end].strip()
    
    # Calculate where the term starts within the fragment
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

    term_end = min(term_start + term_length, len(fragment))
    term = fragment[term_start:term_end]
    before = fragment[:term_start]
    highlighted_term = f"**{term.upper()}**"
    after = fragment[term_end:]
    return before + highlighted_term + after


def get_snippet_for_term(corpus_dir, doc_id, term_positions, fragment_size=15):
    """
    Get a highlighted text snippet for the first occurrence of a term.
    Also supports phrase highlighting when the phrase exists verbatim in fragment.
    
    Parameters:
    - corpus_dir: str path to corpus directory
    - doc_id: str document identifier
    - term_positions: list of character positions
    - fragment_size: int number of words (default: 12)
    
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
        
        first_position = term_positions[0]
        result = extract_snippet(text, first_position, fragment_size)
        
        if result:
            fragment, term_start, term_length = result
            print(fragment[term_start:term_start+term_length])
            print(f"fragment[513]: {fragment[513]}")
            return highlight_term_in_fragment(fragment, term_start, term_length)
        
        return None
    
    except Exception as e:
        return None
