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
from snippet_extractor import get_snippet_for_term
from phrase_searcher import filter_docs_by_phrase


class SearchEngine:
    """
    Search engine that loads and queries a TF-IDF index.
    Coordinates between filtering, query processing, and ranking modules.
    """
    
    def __init__(self, index_path, corpus_dir):
        """
        Initialize search engine.
        
        Parameters:
        - index_path: path to TF-IDF JSON index file
        - corpus_dir: path to corpus documents directory
        """
        self.index_path = index_path
        self.corpus_dir = corpus_dir
        self.index = None
        self.terms = {}
        self.meta = {}
        self.doc_norms = {}
        self.filter = None
        self.ranker = None
        self.phrase_occurrences = {}  # Store phrase occurrences for PHRASE searches
    
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
    
    def get_snippets_for_document(self, doc_id, query_terms):
        """
        Get text snippets for each query term found in document.
        For PHRASE searches, use stored phrase occurrences.
        """
        snippets = {}
        
        # If we have phrase occurrences stored (PHRASE search mode), use first occurrence
        if self.phrase_occurrences and doc_id in self.phrase_occurrences:
            occurrences = self.phrase_occurrences[doc_id]
            if occurrences:
                # Use first occurrence of the phrase
                start_pos, end_pos = occurrences[0]
                snippet = get_snippet_for_term(
                    self.corpus_dir,
                    doc_id,
                    [start_pos],
                    fragment_size=15
                )
                if snippet:
                    # Label as the phrase instead of individual terms
                    phrase_label = ' '.join(query_terms)
                    snippets[phrase_label] = snippet
                return snippets
        
        # Regular mode: snippet per term
        for term in set(query_terms):
            if term in self.terms:
                # Find this document in term's weights
                for weight_entry in self.terms[term]['weights']:
                    if weight_entry['doc'] == doc_id:
                        positions = weight_entry.get('positions', [])
                        snippet = get_snippet_for_term(
                            self.corpus_dir,
                            doc_id,
                            positions,
                            fragment_size=15
                        )
                        if snippet:
                            snippets[term] = snippet
                        break
        
        return snippets
    
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
        
        # Optional PHRASE operator: keep only docs where the full phrase occurs in order
        if operator == 'PHRASE' and len(all_query_terms) > 1:
            candidate_docs, self.phrase_occurrences = filter_docs_by_phrase(self.terms, all_query_terms, candidate_docs)
            print(f"  Phrase filter: {len(candidate_docs)} documents contain the phrase in order")
            if not candidate_docs:
                print("  No documents contain the exact phrase.")
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
        
        # Add snippets to each result
        for doc_info in ranked_docs:
            doc_id = doc_info['doc']
            doc_info['snippets'] = self.get_snippets_for_document(doc_id, all_query_terms)
        
        return ranked_docs