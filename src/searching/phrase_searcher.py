"""
Phrase search module implementing the nextPhrase algorithm with adjacency check.
Finds exact phrase occurrences in documents using term position lists.
"""

ADJACENCY_TOLERANCE = 6  


def positions_for_term_in_doc(terms_dict, term, doc_id):
    """
    Get sorted positions where a term appears in a document.
    """
    for w in terms_dict.get(term, {}).get('weights', []):
        if w['doc'] == doc_id:
            return w.get('positions', [])
    return []


def next_pos(pos_list, v):
    """
    Find the next position in pos_list that is >= v.
    """
    for p in pos_list:
        if p >= v:
            return p
    return float('inf')


def prev_pos(pos_list, u):
    """
    Find the previous position in pos_list that is <= u.
    """
    prev = float('inf')
    for p in pos_list:
        if p <= u:
            prev = p
        else:
            break
    return prev if prev != float('inf') else float('inf')


def check_adjacency(positions, query_terms):
    """
    Verify if the terms in the found positions are actually adjacent.
    
    Parameters:
    - positions: list of integer positions found for the terms
    - query_terms: list of the terms (strings) to check lengths
    
    Returns:
    - bool: True if terms are adjacent (close enough), False otherwise
    """
    for i in range(len(positions) - 1):
        current_pos = positions[i]
        next_pos = positions[i+1]
        term_len = len(query_terms[i])
        
        expected_next_min = current_pos + term_len
        
        distance = next_pos - expected_next_min
        
        if distance > ADJACENCY_TOLERANCE:
            return False
            
    return True


def next_phrase(terms_dict, query_terms, doc_id, position):
    """
    nextPhrase(t1, t2, ..., tn, position)
    
    Find the next occurrence of the EXACT phrase (adjacent words).
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
        if check_adjacency(backward_positions, query_terms):
            return (backward_positions[0], backward_positions[-1])
        else:
            return next_phrase(terms_dict, query_terms, doc_id, backward_positions[0] + 1)
            
    else:
        return next_phrase(terms_dict, query_terms, doc_id, backward_positions[0] + 1)
    

def all_phrase_occurrences(terms_dict, query_terms, doc_id):
    """
    Find ALL occurrences of the exact phrase in a document.
    """
    occurrences = []
    u = 0
    
    while u != float('inf'):
        result = next_phrase(terms_dict, query_terms, doc_id, u)
        if result is None:
            break
        
        start_pos, end_pos = result
        occurrences.append((start_pos, end_pos))
        u = end_pos + 1
    
    return occurrences


def filter_docs_by_phrase(terms_dict, query_terms, candidate_docs):
    """
    Filter documents to keep only those containing the exact phrase.
    """
    if len(query_terms) <= 1:
        return candidate_docs, {}
    
    filtered = set()
    occurrences_map = {}
    
    for doc_id in candidate_docs:
        occurrences = all_phrase_occurrences(terms_dict, query_terms, doc_id)
        if occurrences:
            filtered.add(doc_id)
            occurrences_map[doc_id] = occurrences
    
    return filtered, occurrences_map