"""
Text extraction utilities for searching and snippet generation.
"""
import os
import re

from config import CORPUS_DATA_DIR

# PUNCTUATION MARKS for sentence boundary detection
PUNCTUATION_MARKS = {'.', ',', ';', ':', '(', ')', '[', ']', '-', '"', '¿', '?', '¡', '!'}

def highlight_term(text, term):
    """
    Highlight a search term in text by wrapping it with ** markers (markdown bold).
    Case-insensitive matching.
    
    Parameters:
    - text: str, the text to highlight
    - term: str, the term to highlight
    
    Returns:
    - str: text with term highlighted as **term**
    """
    if not text or not term:
        return text
    
    # Simple case-insensitive replacement
    pattern = re.escape(term)
    highlighted = re.sub(pattern, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)
    return highlighted


def highlight_phrase(text, phrase):
    """
    Highlight a complete phrase in text by wrapping it with ** markers (markdown bold).
    Treats the entire phrase as a single unit, case-insensitive matching.
    
    Parameters:
    - text: str, the text to highlight
    - phrase: str, the complete phrase to highlight (e.g., "hola amigos")
    
    Returns:
    - str: text with phrase highlighted as **phrase**
    """
    if not text or not phrase:
        return text
    
    # Escape special regex characters in the phrase
    pattern = re.escape(phrase)
    # Replace with highlighted version as a single unit
    highlighted = re.sub(pattern, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)
    return highlighted
    
def get_snippet_for_term(doc_id, position):
    """
    Extract a sentence/phrase from a document containing the word at a specific position.
    Finds the nearest punctuation marks (forward and backward) to define sentence boundaries.
    
    Parameters:
    - doc_id: str, document identifier (filename)
    - position: int, character position in the document
    
    Returns:
    - str: extracted text fragment (sentence/phrase) containing the position
    """
    text = get_full_document_text(doc_id)
    
    start_pos = position
    end_pos = position

    # Search backward from position
    while start_pos > 0:
        start_pos -= 1
        if text[start_pos] in PUNCTUATION_MARKS:
            # If it's a hyphen between alphanumeric chars
            if text[start_pos] == '-' and start_pos > 0 and start_pos + 1 < len(text):
                if text[start_pos - 1].isalnum() and text[start_pos + 1].isalnum():
                    continue
            start_pos += 1  # Move past the punctuation mark
            break
    
    # Skip any whitespace after the punctuation
    while start_pos < len(text) and text[start_pos] == ' ':
        start_pos += 1
    
    # Search forward from position
    while end_pos < len(text):
        if text[end_pos] in PUNCTUATION_MARKS:
            # If it's a hyphen followed by alphanumeric char
            if text[end_pos] == '-' and end_pos + 1 < len(text) and text[end_pos + 1].isalnum():
                end_pos += 1
                continue
            break
        end_pos += 1
        
    word_counter = len(text[start_pos:end_pos].split())
    
    if not ((word_counter <= 15) and (word_counter >= 10)):
        # Reallocate positions to get approximately 15 words
        if (position - start_pos) > (end_pos - position):
            start_pos = reallocate_position(text, end_pos, 15, forward=False, backward=True)
        else:
            end_pos = reallocate_position(text, start_pos, 15, forward=True, backward=False)

    # Extract the snippet
    return extract_text_by_position(doc_id, start_pos, end_pos)

def extract_text_by_position(doc_id, start_pos, end_pos):
    """
    Extract text from a document based on character positions.
    
    Parameters:
    - doc_id: str, document identifier (filename)
    - start_pos: int, starting character position (0-indexed)
    - end_pos: int, ending character position (exclusive)
    
    Returns:
    - str: extracted text fragment, or None if document not found
    """
    doc_path = os.path.join(CORPUS_DATA_DIR, doc_id)
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_id}")
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Validate positions
        if start_pos < 0:
            start_pos = 0
        if end_pos > len(content):
            end_pos = len(content)
        if start_pos >= end_pos:
            return ""
        
        # Extract the text fragment
        extracted_text = content[start_pos:end_pos]
        return extracted_text
    
    except Exception as e:
        raise IOError(f"Error reading document {doc_id}: {str(e)}")


def reallocate_position(text, position, num_steps, forward, backward):
    """
    Reallocate position to the nearest word boundary in the specified direction.
    
    Parameters:
    - text: str, the document text
    - position: int, current character position
    - num_steps: int, number of word boundaries to move
    - forward: bool, whether to move forward
    - backward: bool, whether to move backward
    
    Returns:
    - int: new position at the nearest word boundary
    """
    # Move position forward to the nearest word boundary
    if forward:
        while position < len(text) and num_steps > 0:
            if text[position] in {' ', '\n', '\t'}:
                num_steps -= 1
            position += 1
    
    # Move position backward to the nearest word boundary
    if backward:
        while position > 0 and num_steps > 0:
            if text[position - 1] in {' ', '\n', '\t'}:
                num_steps -= 1
            position -= 1

    return position

def find_snippet_for_term(doc_id, term):
    """
    Find and extract a snippet from the document containing the specified term.
    Highlights the found term in the result.
    
    Parameters:
    - doc_id: str, document identifier (filename)
    - term: str, search term
    
    Returns:
    - str: extracted text fragment containing the term with the term highlighted as **term**
    """
    text = get_full_document_text(doc_id)
    position = text.find(term)

    if position == -1:
        return None

    return get_snippet_for_term(doc_id, position)

def get_full_document_text(doc_id):
    """
    Retrieve the complete text of a document.
    
    Parameters:
    - doc_id: str, document identifier (filename)
    
    Returns:
    - str: complete document text
    """
    doc_path = os.path.join(CORPUS_DATA_DIR, doc_id)
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_id}")
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except Exception as e:
        raise IOError(f"Error reading document {doc_id}: {str(e)}")

def extract_snippet(doc_id, position, term, searched_term, operator):
    """
    Extract a snippet from the document based on position or term.
    
    Parameters:
    - doc_id: str, document identifier (filename)
    - position: int, character position in the document
    - term: str, search term or phrase
    - searched_term: str, the original searched term(s)
    - operator: str, 'AND', 'OR', or 'PHRASE'
    
    Returns:
    - str: extracted text fragment (sentence/phrase) containing the position
    """
    if (position != -1):
        snippet = get_snippet_for_term(doc_id, position)
    else:
        snippet = find_snippet_for_term(doc_id, term)

    # For PHRASE operator, highlight the complete phrase as a unit
    if operator == 'PHRASE':
        return highlight_phrase(snippet, searched_term)
    else:
        return highlight_term(snippet, searched_term)
    