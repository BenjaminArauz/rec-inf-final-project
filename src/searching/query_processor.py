"""
Query processing module for calculating TF-IDF norms and weights.
"""
import math


def compute_query_norm(query_terms, index_terms):
    """
    Compute the vector norm for the query.
    Formula: sqrt(sum(IDF^2)) for each unique term in query
    
    Parameters:
    - query_terms: list of preprocessed query terms
    - index_terms: dict with term data from TF-IDF index
    
    Returns:
    - float: query vector norm
    """
    # Get unique terms from query
    unique_terms = set(query_terms)
    
    sum_squares = 0.0
    for term in unique_terms:
        if term in index_terms:
            # Get IDF from index
            idf = index_terms[term]['idf']
            # Add IDF squared
            sum_squares += idf ** 2
    
    return math.sqrt(sum_squares)
