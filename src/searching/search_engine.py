"""
Search engine module for querying the TF-IDF index.
Orchestrates document filtering, query processing, and ranking.
"""
import json
import os
import sys

# Path adjustment for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MAX_RESULTS
from indexing.cleaner import preprocess_text
from document_filter import DocumentFilter
from document_ranker import DocumentRanker
from text_extractor import extract_snippet
from phrase_searcher import filter_docs_by_phrase


class SearchEngine:
    """
    Search engine that loads and queries a TF-IDF index.
    Coordinates between filtering, query processing, and ranking modules.
    """
    
    def __init__(self, index_path, corpus_dir, max_results=None):
        """
        Initialize search engine.
        
        Parameters:
        - index_path: path to TF-IDF JSON index file
        - corpus_dir: path to corpus documents directory
        - max_results: int, maximum number of results to return (uses config default if None)
        """
        self.index_path = index_path
        self.corpus_dir = corpus_dir
        self.max_results = max_results if max_results is not None else MAX_RESULTS
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
    
    def is_document_matching(self, doc_id, term):
        """
        Check if a document contains a specific term and retrieve positions.
        Parameters:
        - doc_id: str document identifier
        - term: str search term
        Returns:
        - tuple: (bool is_found, list positions)
        """
        if term in self.terms:
            for doc_info in self.terms[term]['weights']:
                if doc_info['doc'] == doc_id:
                    return True, doc_info['positions']
            
        return False, []

    def get_positions(self, doc_id, query_terms, original_query, operator='AND'):
        """
        Extract snippets for a document based on query terms.
        
        Parameters:
        - doc_id: str document identifier
        - query_terms: list of str query terms
        - original_query: list of str original query words
        - operator: str 'AND', 'OR', 'PHRASE' (default 'AND')
        
        Returns:
        - list: snippets for the document
        """
        snippets = []
        # For PHRASE operator, treat all terms as a single phrase
        if operator == 'PHRASE' and len(query_terms) > 1:
            # Build the full phrase
            full_phrase = ' '.join(query_terms)
            term = query_terms[0]  # Use first term to find position
            is_found, positions = self.is_document_matching(doc_id, term)
            if is_found:
                snippet = extract_snippet(doc_id, positions[0], full_phrase, ' '.join(original_query), operator)
                snippets.append(snippet)
        else:
            # For AND/OR, process each term individually
            for i in range(len(query_terms)):
                term = query_terms[i]
                is_found, positions = self.is_document_matching(doc_id, term)
                if is_found:
                    snippet = extract_snippet(doc_id, positions[0], term, original_query[i], operator)
                    snippets.append(snippet)
        return snippets

    def search(self, query, operator):
        """
        Search for documents matching the query and rank by similarity.
        
        Process:
        1. Preprocess query terms
        2. Filter documents based on operator (AND/OR)
        3. Compute query vector norm
        4. Rank filtered documents by cosine similarity
        
        Parameters:
        - query: str search terms (space-separated, without operators)
        - operator: str 'AND', 'OR', 'PHRASE'
        
        Returns:
        - list: ranked documents with similarity scores
        """
        if not self.index:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        print(f"\nQuery processing:")
        print(f"  Search terms: '{query}'")
        print(f"  Operator: {operator}")
        
        # Preprocess query to extract terms
        all_query_terms = self.preprocess_query(query)
        print(f"  Processed terms: {all_query_terms}")
        
        if not all_query_terms:
            print("No valid terms in query after preprocessing.")
            return []
        
        # Filter documents based on operator
        
        candidate_docs = self.filter.filter_documents(all_query_terms, operator)
        print(f"  Filter: {operator} - {len(candidate_docs)} documents matched")

        # Optional PHRASE operator: keep only docs where the full phrase occurs in order
        if operator == 'PHRASE' and len(all_query_terms) > 1:
            candidate_docs, self.phrase_occurrences = filter_docs_by_phrase(self.terms, all_query_terms, candidate_docs)
            print(f"  Phrase filter: {len(candidate_docs)} documents contain the phrase in order")
            
        if not candidate_docs:
            print("  No documents found matching the criteria.")
            return []
        
        # Compute query vector norm
        query_norm = self.ranker.compute_query_norm(all_query_terms, self.terms)
        
        # Rank documents by cosine similarity
        ranked_docs = self.ranker.rank_documents(all_query_terms, query_norm, candidate_docs)
        
        if not ranked_docs:
            print("  No documents found containing query terms.")
            return []
        
        # Limit results to max_results (top n documents)
        top_docs = ranked_docs[:self.max_results]
        
        # Add snippets to each result
        for doc_info in top_docs:
            doc_info['snippets'] = self.get_positions(doc_info['doc'], all_query_terms, query.split(), operator)
            
        return top_docs