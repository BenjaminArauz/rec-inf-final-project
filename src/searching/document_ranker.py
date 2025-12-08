"""
Document ranking module for similarity calculations.
"""


class DocumentRanker:
    """
    Ranks documents based on cosine similarity with query.
    """
    
    def __init__(self, index_terms, doc_norms):
        """
        Initialize document ranker.
        
        Parameters:
        - index_terms: dict with term data from TF-IDF index
        - doc_norms: dict with document vector norms
        """
        self.index_terms = index_terms
        self.doc_norms = doc_norms
    
    def rank_documents(self, query_terms, query_norm, candidate_docs=None):
        """
        Rank documents by cosine similarity with the query.
        
        Formula for each document:
        similarity = Σ(IDF × TF-IDF_doc) / (doc_norm × query_norm)
        
        Parameters:
        - query_terms: list of preprocessed query terms
        - query_norm: float, vector norm of the query
        - candidate_docs: optional set of document IDs to rank (for AND/OR filtering)
        
        Returns:
        - list: ranked documents with similarity scores, sorted descending
        """
        # Get unique terms from query
        unique_query_terms = set(query_terms)
        
        if query_norm == 0:
            return []
        
        # Dictionary to accumulate numerator for each document
        doc_numerators = {}
        
        # For each term in the query
        for term in unique_query_terms:
            if term not in self.index_terms:
                continue
            
            idf = self.index_terms[term]['idf']
            
            # For each document containing this term
            for weight_entry in self.index_terms[term]['weights']:
                doc_id = weight_entry['doc']
                
                # If candidate_docs is specified, only rank those documents
                if candidate_docs is not None and doc_id not in candidate_docs:
                    continue
                
                tfidf_in_doc = weight_entry['tfidf']
                
                # Calculate contribution: IDF × TF-IDF in doc
                contribution = idf * tfidf_in_doc
                
                # Accumulate for this document
                if doc_id not in doc_numerators:
                    doc_numerators[doc_id] = 0.0
                doc_numerators[doc_id] += contribution
        
        if not doc_numerators:
            return []
        
        # Calculate final similarity scores
        doc_scores = []
        
        for doc_id, numerator in doc_numerators.items():
            # Denominator: doc_norm × query_norm
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            
            if doc_norm == 0:
                continue
            
            similarity = numerator / (doc_norm * query_norm)
            
            doc_scores.append({
                'doc': doc_id,
                'score': similarity
            })
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return doc_scores
