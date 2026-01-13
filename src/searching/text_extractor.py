"""
Text extraction utilities for searching and snippet generation.
"""
import os
import re
import sys

# Path adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CORPUS_DATA_DIR

SNIPPET_SIZE = 15 

def highlight_term(text, term):
    """Highlight a search term in text."""
    if not text or not term:
        return text
    pattern = re.escape(term)
    return re.sub(pattern, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)

def highlight_phrase(text, phrase):
    """Highlight a complete phrase in text."""
    if not text or not phrase:
        return text
    pattern = re.escape(phrase)
    return re.sub(pattern, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)

def get_snippet_for_term(doc_id, position):
    """
    Extract a snippet centered on the position, ignoring sentence boundaries
    if they are too far away, to ensure the term is visible.
    """
    text = get_full_document_text(doc_id)
    
    words_before = SNIPPET_SIZE // 2
    words_after = SNIPPET_SIZE - words_before
    
    start_pos = position
    end_pos = position

    count = 0
    while start_pos > 0 and count < words_before:
        start_pos -= 1
        if text[start_pos] == ' ' and start_pos + 1 < len(text) and text[start_pos+1] != ' ':
            count += 1
            
    end_pos = position
    count = 0
    while end_pos < len(text) and count < words_after:
        end_pos += 1
        if end_pos < len(text) and text[end_pos] == ' ':
            count += 1

    while start_pos > 0 and text[start_pos] != ' ':
        start_pos -= 1
    while end_pos < len(text) and text[end_pos] != ' ':
        end_pos += 1
        
    return extract_text_by_position(doc_id, start_pos, end_pos).strip()

def extract_text_by_position(doc_id, start_pos, end_pos):
    """Extract text from a document based on character positions."""
    doc_path = os.path.join(CORPUS_DATA_DIR, doc_id)
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_id}")
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if start_pos < 0: start_pos = 0
        if end_pos > len(content): end_pos = len(content)
        return content[start_pos:end_pos]
    except Exception as e:
        raise IOError(f"Error reading document {doc_id}: {str(e)}")

def find_snippet_for_term(doc_id, term):
    """Find snippet searching for term string."""
    text = get_full_document_text(doc_id)
    position = text.find(term)
    if position == -1: return None
    return get_snippet_for_term(doc_id, position)

def get_full_document_text(doc_id):
    """Retrieve complete text."""
    doc_path = os.path.join(CORPUS_DATA_DIR, doc_id)
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_id}")
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except Exception as e:
        raise IOError(f"Error reading document {doc_id}: {str(e)}")

def extract_snippet(doc_id, position, term, searched_term, operator):
    """Orchestrator for extraction."""
    if position != -1:
        snippet = get_snippet_for_term(doc_id, position)
    else:
        snippet = find_snippet_for_term(doc_id, term)

    if snippet:
        snippet = snippet.replace('\n', ' ')
        if operator == 'PHRASE':
            return highlight_phrase(snippet, searched_term)
        else:
            return highlight_term(snippet, searched_term)
    return ""