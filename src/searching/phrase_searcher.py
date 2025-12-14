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


def next_phrase(terms_dict, query_terms, doc_id, position):
    """
    nextPhrase(t1, t2, ..., tn, position)
    
    Find the next occurrence of the exact phrase (query_terms in order) in a document.
    Implements the nextPhrase algorithm exactly as specified in the pseudocode.
    
    Parameters:
    - terms_dict: dict from TF-IDF index with all term data
    - query_terms: list of terms forming the phrase [t1, t2, ..., tn]
    - doc_id: str document identifier
    - position: int starting position for search (default: 0)
    
    Returns:
    - tuple (u, v) if phrase found, None otherwise
    """
    n = len(query_terms)
    
    v = position
    
    forward_positions = []
    for i in range(n):
        pos_list = positions_for_term_in_doc(terms_dict, query_terms[i], doc_id)
        v = next_pos(pos_list, v)
        if v == float('inf'):
            return None
        forward_positions.append(v)
    
    u = v
    
    backward_positions = []
    for i in range(n - 1, -1, -1):
        pos_list = positions_for_term_in_doc(terms_dict, query_terms[i], doc_id)
        u = prev_pos(pos_list, u)
        if u == float('inf'):
            return None
        backward_positions.insert(0, u)

    if forward_positions == backward_positions:
        return (backward_positions[0], backward_positions[-1])
    else:
        return next_phrase(terms_dict, query_terms, doc_id, backward_positions[0] + 1)
    

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
        result = next_phrase(terms_dict, query_terms, doc_id, u)
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
