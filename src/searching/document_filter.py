"""
Document filtering module for AND/OR operations.
"""


class DocumentFilter:
    """
    Handles document filtering based on query terms and operators.
    """
    
    def __init__(self, index_terms):
        """
        Initialize document filter.
        
        Parameters:
        - index_terms: dict with term data from TF-IDF index
        """
        self.index_terms = index_terms
    
    def get_documents_for_terms(self, terms):
        """
        Get all documents that contain the given terms.
        
        Parameters:
        - terms: list of preprocessed terms
        
        Returns:
        - set: document IDs containing at least one term
        """
        docs = set()
        for term in terms:
            if term in self.index_terms:
                for weight_entry in self.index_terms[term]['weights']:
                    docs.add(weight_entry['doc'])
        return docs
    
    def filter_and(self, terms):
        """
        Filter documents using AND operator (documents must contain ALL terms).
        
        Parameters:
        - terms: list of preprocessed terms
        
        Returns:
        - set: document IDs that contain all query terms
        """
        if not terms:
            return set()
        
        # Start with documents containing the first term
        first_term = terms[0]
        if first_term not in self.index_terms:
            return set()
        
        result_docs = self.get_documents_for_terms([first_term])
        
        # Intersect with documents containing each subsequent term
        for term in terms[1:]:
            if term in self.index_terms:
                term_docs = self.get_documents_for_terms([term])
                result_docs = result_docs.intersection(term_docs)
            else:
                # Term not in index, no documents can match
                return set()
        
        return result_docs
    
    def filter_or(self, terms):
        """
        Filter documents using OR operator (documents must contain AT LEAST ONE term).
        
        Parameters:
        - terms: list of preprocessed terms
        
        Returns:
        - set: document IDs that contain at least one query term
        """
        return self.get_documents_for_terms(terms)
    
    def filter_documents(self, terms, operator='OR'):
        """
        Filter documents based on operator.
        
        Parameters:
        - terms: list of preprocessed terms
        - operator: 'AND' or 'OR'
        
        Returns:
        - set: filtered document IDs
        """
        if operator == 'AND':
            return self.filter_and(terms)
        else:  # OR or default
            return self.filter_or(terms)
