"""
Search engine module for querying the TF-IDF index.
Orchestrates document filtering, query processing, and ranking.
"""
import json
import os
import sys

# Path adjustment for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indexing.cleaner import preprocess_text
from document_filter import DocumentFilter
from query_processor import compute_query_norm
from document_ranker import DocumentRanker


class SearchEngine:
    """
    Search engine that loads and queries a TF-IDF index.
    Coordinates between filtering, query processing, and ranking modules.
    """
    
    def __init__(self, index_path):
        """
        Initialize search engine.
        
        Parameters:
        - index_path: path to TF-IDF JSON index file
        """
        self.index_path = index_path
        self.index = None
        self.terms = {}
        self.meta = {}
        self.doc_norms = {}
        self.filter = None
        self.ranker = None
    
    def load_index(self):
        """
        Load TF-IDF index from JSON file and initialize helper modules.
        
        Returns:
        - bool: True if loaded successfully
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"TF-IDF index not found at: {self.index_path}")
        
        with open(self.index_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        
        self.meta = self.index.get('meta', {})
        self.terms = self.index.get('terms', {})
        self.doc_norms = self.index.get('doc_norms', {})
        
        self.filter = DocumentFilter(self.terms)
        self.ranker = DocumentRanker(self.terms, self.doc_norms)
        
        return True
    
    def get_index_info(self):
        """
        Get metadata about the loaded index.
        
        Returns:
        - dict: index metadata
        """
        return {
            'total_docs': self.meta.get('total_docs', 0),
            'total_terms': self.meta.get('total_terms', 0),
            'index_loaded': self.index is not None
        }
    
    def preprocess_query(self, query):
        """
        Apply the same preprocessing to query as was done to documents.
        
        Parameters:
        - query: str raw query
        
        Returns:
        - list: preprocessed query terms
        """
        query_terms, _ = preprocess_text(query)
        return query_terms
    
    def search(self, query, operator='OR'):
        """
        Search for documents matching the query and rank by similarity.
        
        Process:
        1. Preprocess query terms
        2. Filter documents based on operator (AND/OR)
        3. Compute query vector norm
        4. Rank filtered documents by cosine similarity
        
        Parameters:
        - query: str search terms (space-separated, without operators)
        - operator: str 'AND' or 'OR' (default: 'OR')
        
        Returns:
        - list: ranked documents with similarity scores
        """
        if not self.index:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        print(f"\nQuery processing:")
        print(f"  Search terms: '{query}'")
        print(f"  Operator: {operator}")
        
        # Step 1: Preprocess query to extract terms
        all_query_terms = self.preprocess_query(query)
        print(f"  Processed terms: {all_query_terms}")
        
        if not all_query_terms:
            print("No valid terms in query after preprocessing.")
            return []
        
        # Step 2: Filter documents based on operator
        candidate_docs = self.filter.filter_documents(all_query_terms, operator)
        print(f"  Filter: {operator} - {len(candidate_docs)} documents matched")
        
        if not candidate_docs:
            print("  No documents found matching the criteria.")
            return []
        
        # Step 3: Compute query vector norm
        query_norm = compute_query_norm(all_query_terms, self.terms)
        
        # Step 4: Rank documents by cosine similarity
        ranked_docs = self.ranker.rank_documents(
            all_query_terms, 
            query_norm, 
            candidate_docs
        )
        
        if not ranked_docs:
            print("  No documents found containing query terms.")
            return []
        
        # Display results
        print(f"\n  Found {len(ranked_docs)} document(s):")
        for i, doc_info in enumerate(ranked_docs, 1):
            print(f"    {i}. Document: {doc_info['doc']}")
            print(f"       Similarity: {doc_info['score']:.4f}")
        
        return ranked_docs
