"""
Indexer module for processing documents and computing TF-IDF matrix.
"""
import os
import math
from collections import defaultdict
from cleaner import compute_tf
from storage import save_json


class DocumentIndexer:
    """
    Handles document processing and TF-IDF matrix construction.
    """
    
    def __init__(self, corpus_dir):
        """
        Initialize the indexer.
        
        Parameters:
        - corpus_dir: path to the directory containing documents
        """
        self.corpus_dir = corpus_dir
        self.tf_matrix = defaultdict(list)  
        self.tfidf_matrix = {}
        self.doc_norms = {}
        self.total_docs = 0
    
    def build_index(self):
        """Build the complete TF-IDF index by processing all documents."""
        if not os.path.exists(self.corpus_dir):
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        
        files = os.listdir(self.corpus_dir)
        index = 0
        
        # Step 1: Process all documents and collect TF values
        for filename in files:
            filepath = os.path.join(self.corpus_dir, filename)
            
            if os.path.isfile(filepath):
                self.process_single_document(filename, filepath)
                index += 1
        
        self.total_docs = index
        
        # Step 2: Compute TF-IDF matrix
        self.compute_tfidf()
        
        return self.tfidf_matrix
    
    def process_single_document(self, filename, filepath):
        """
        Process a single document and update TF matrix with positions.
        
        Parameters:
        - filename: name of the file
        - filepath: full path to the file
        """
        print(f"Processing file: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Compute TF (always includes character positions)
        tf_data = compute_tf(content)
        
        # Store TF values with positions indexed by document
        for term, term_info in tf_data.items():
            self.tf_matrix[term].append({
                'doc': filename,
                'tf': term_info['tf'],
                'positions': term_info['positions']
            })
    
    def compute_tfidf(self):
        """
        Compute IDF, TF-IDF weights, and vector norms per document.
        """
        # Dictionary to accumulate squared TF-IDF values per document
        doc_squared_sums = defaultdict(float)
        
        for term, tf_list in self.tf_matrix.items():
            # Calculate IDF: log2(total_docs / num_docs_with_term)
            idf = math.log2(self.total_docs / len(tf_list))
            
            # Calculate TF-IDF weights
            weight_values = []
            
            for doc_entry in tf_list:
                doc_id = doc_entry['doc']
                tf_val = doc_entry['tf']
                positions = doc_entry['positions']
                
                weight = tf_val * idf
                weight_values.append({
                    'doc': doc_id,
                    'tfidf': weight,
                    'positions': positions
                })
                
                # Accumulate squared weight for this document
                doc_squared_sums[doc_id] += weight ** 2
            
            # Store: [idf, [{doc, tfidf, positions}, ...]]
            self.tfidf_matrix[term] = [idf, weight_values]
        
        # Calculate vector norm for each document
        for doc_id, sum_squares in doc_squared_sums.items():
            self.doc_norms[doc_id] = math.sqrt(sum_squares)

    def to_serializable(self):
        """
        Build a JSON-serializable dict with metadata, TF-IDF data, and document norms.
        Terms are sorted alphabetically.
        """
        terms = {}
        # Sort terms alphabetically
        for term in sorted(self.tfidf_matrix.keys()):
            data = self.tfidf_matrix[term]
            idf, weights = data
            terms[term] = {
                "idf": idf,
                "weights": [
                    {
                        "doc": w['doc'],
                        "tfidf": w['tfidf'],
                        "positions": w['positions']
                    } for w in weights
                ]
            }
        return {
            "meta": {
                "total_docs": self.total_docs,
                "total_terms": len(self.tfidf_matrix),
            },
            "terms": terms,
            "doc_norms": self.doc_norms
        }

    def save_json(self, output_path, ensure_ascii=True):
        """Serialize current TF-IDF matrix and persist it to JSON."""
        payload = self.to_serializable()
        save_json(payload, output_path, ensure_ascii=ensure_ascii)
        return output_path
    
    def get_results(self):
        """
        Get TF-IDF matrix (internal structure).
        """
        return self.tfidf_matrix
    
    def print_summary(self):
        """
        Print a summary of the indexing results.
        
        Parameters:
        - sample_size: number of sample terms to display
        """
        print("=" * 60)
        print("INDEXING SUMMARY")
        print("=" * 60)
        print(f"Total documents processed: {self.total_docs}")
        print(f"Total unique terms: {len(self.tfidf_matrix)}")

        for term, data in self.tf_matrix.items():
            print(f"Term: '{term}' TFs: {data}")

        print("-" * 60)
        
        for term, data in self.tfidf_matrix.items():
            print(f"Term: '{term}' {data}")
        print("=" * 60)
