"""
Phrase search module implementing the nextPhrase algorithm.
Finds exact phrase occurrences in documents using term position lists.
"""


def positions_for_term_in_doc(terms_dict, term, doc_id):
    """
    Get sorted positions where a term appears in a document.
    
    Parameters:
    - terms_dict: dict from TF-IDF index with term data
    - term: str the term to search for
    - doc_id: str document identifier
    
    Returns:
    - list: sorted character positions or empty list
    """
    for w in terms_dict.get(term, {}).get('weights', []):
        if w['doc'] == doc_id:
            return w.get('positions', [])
    return []


def next_pos(pos_list, v):
    """
    Find the next position in pos_list that is >= v.
    
    Parameters:
    - pos_list: list of sorted positions
    - v: int current position
    
    Returns:
    - int next position or float('inf') if not found
    """
    for p in pos_list:
        if p >= v:
            return p
    return float('inf')


def prev_pos(pos_list, u):
    """
    Find the previous position in pos_list that is <= u.
    
    Parameters:
    - pos_list: list of sorted positions
    - u: int current position
    
    Returns:
    - int previous position or float('inf') if not found
    """
    prev = float('inf')
    for p in pos_list:
        if p <= u:
            prev = p
        else:
            break
    return prev if prev != float('inf') else float('inf')


def next_phrase(terms_dict, query_terms, doc_id, start_v=0):
    """
    Find the next occurrence of the exact phrase (query_terms in order) in a document.
    Implements the nextPhrase algorithm from the specification (recursive version).
    
    Algorithm:
    1. Forward pass: find next occurrence of each term starting from position start_v
    2. Backward pass: verify positions are consecutive
    3. If valid, return (u, v); else recursively search from next position
    
    Parameters:
    - terms_dict: dict from TF-IDF index with all term data
    - query_terms: list of terms forming the phrase, in order
    - doc_id: str document identifier
    - start_v: int starting position for search (default: 0)
    
    Returns:
    - tuple (start_pos, end_pos) if phrase found, None otherwise
      start_pos: character position where first term starts
      end_pos: character position where last term starts
    """
    if not query_terms:
        return None
    
    # Forward pass: find next occurrence of each term
    found_positions = []
    v = start_v
    
    for t in query_terms:
        pos_list = positions_for_term_in_doc(terms_dict, t, doc_id)
        v = next_pos(pos_list, v)
        if v == float('inf'):
            # No more occurrences of this term, phrase not found
            return None
        found_positions.append(v)
        v += 1  # Move past this occurrence for next term search
    
    # found_positions = [pos_t1, pos_t2, ..., pos_tn]
    u = found_positions[0]
    v = found_positions[-1]
    
    # Backward pass: verify each term is at the expected position
    valid = True
    expected_pos_idx = len(query_terms) - 1
    check_v = v
    
    for t in reversed(query_terms):
        pos_list = positions_for_term_in_doc(terms_dict, t, doc_id)
        found_pos = prev_pos(pos_list, check_v)
        
        if found_pos != found_positions[expected_pos_idx]:
            # Position mismatch, phrase is not consecutive
            valid = False
            break
        
        check_v = found_pos - 1
        expected_pos_idx -= 1
    
    # If positions are consecutive, return the phrase occurrence
    if valid:
        return (u, v)
    else:
        # Try next occurrence recursively from u + 1
        return next_phrase(terms_dict, query_terms, doc_id, u + 1)


def all_phrase_occurrences(terms_dict, query_terms, doc_id):
    """
    Find ALL occurrences of the exact phrase in a document.
    Implements the outer loop from the specification to find all phrase occurrences.
    
    Parameters:
    - terms_dict: dict from TF-IDF index with all term data
    - query_terms: list of terms forming the phrase, in order
    - doc_id: str document identifier
    
    Returns:
    - list: list of tuples (start_pos, end_pos) for each occurrence, or empty list
    """
    occurrences = []
    u = 0
    
    while u != float('inf'):
        result = next_phrase(terms_dict, query_terms, doc_id, start_v=u)
        if result is None:
            break
        
        start_pos, end_pos = result
        occurrences.append((start_pos, end_pos))
        # Continue searching from position after this occurrence
        u = end_pos + 1
    
    return occurrences


def filter_docs_by_phrase(terms_dict, query_terms, candidate_docs):
    """
    Filter documents to keep only those containing the exact phrase.
    Also returns a mapping of doc_id -> list of phrase occurrences.
    
    Parameters:
    - terms_dict: dict from TF-IDF index with all term data
    - query_terms: list of terms forming the phrase, in order
    - candidate_docs: set of document IDs to filter
    
    Returns:
    - tuple: (set of doc_ids, dict of doc_id -> list of occurrences)
      occurrences format: list of (start_pos, end_pos) tuples
    """
    if len(query_terms) <= 1:
        # Single term or empty, all docs pass
        return candidate_docs, {}
    
    filtered = set()
    occurrences_map = {}
    
    for doc_id in candidate_docs:
        occurrences = all_phrase_occurrences(terms_dict, query_terms, doc_id)
        if occurrences:
            filtered.add(doc_id)
            occurrences_map[doc_id] = occurrences
    
    return filtered, occurrences_map
