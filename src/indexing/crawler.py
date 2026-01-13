"""
Advanced Web Crawler with Breadth-First Search (BFS)
Based on WebCrawler architecture with modular components
"""
import sys
import os
from collections import deque

# Import crawler utilities
from crawler_utils import fetch, LinkExtractor

# Import config values
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (CRAWLER_MAX_PAGES, CRAWLER_MAX_DEPTH, CRAWLER_DOMAIN_LIMIT, 
                    CRAWLER_TIMEOUT, CRAWLER_KEYWORDS, CORPUS_DATA_DIR)

class WebCrawler:
    """
    Main crawler class implementing Breadth-First Search (BFS)
    """
    
    def __init__(self):
        """
        Initialize crawler state and components
        """
        self.visited = set()
        self.to_visit = deque()
        self.results = []
        
        # Initialize components
        self.timeout = CRAWLER_TIMEOUT
        self.extractor = LinkExtractor(CRAWLER_DOMAIN_LIMIT)
    
    def should_crawl(self, url, title):
        """
        Determine if a URL should be crawled based on keywords
        
        Parameters:
        - url: str, URL to evaluate
        - title: str, page title

        Returns:
        - bool: True if should crawl

        """
        keywords = CRAWLER_KEYWORDS
        if not keywords:
            return True
        
        keywords_lower = [k.lower() for k in keywords]
        text = (url + ' ' + title).lower()
        return any(k in text for k in keywords_lower)
    
    def is_corpus_document(self, url):
        """
        Check if URL points to a corpus document
        
        Parameters:
        - url: str, URL to check

        Returns:
        - bool: True if corpus document
        """
        return '/corpus/' in url
    
    def download_document(self, url, response):
        """
        Download and save document from URL if not already present

        Parameters:
        - url: str, URL of the document
        - response: requests.Response, HTTP response object

        Returns:
        - bool: True if downloaded, False if skipped or error
        """
        dest_dir = CORPUS_DATA_DIR
        if not dest_dir:
            return False
        
        # Extract document ID from URL
        doc_id = url.rstrip('/').split('/')[-1]
        dest_path = os.path.join(dest_dir, doc_id)
        
        # Skip if already exists
        if os.path.exists(dest_path):
            print(f"  [SKIPPED] {doc_id} (already exists)")
            return False
        
        try:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            print(f"  [DOWNLOADED] {doc_id}")
            return True
        except Exception as e:
            print(f"  [DOWNLOAD ERROR] {doc_id}: {e}")
            return False
    
    def crawl(self, start_url):
        """
        Main crawling loop using Breadth-First Search (BFS).

        Parameters:
        - start_url: str, starting URL for the crawl
        
        Returns:
        - list: Crawl results
        """        
        # Initialize BFS queue with start URL at depth 0
        self.to_visit.append((start_url, 0))
        
        # BFS loop
        while self.to_visit and len(self.results) < CRAWLER_MAX_PAGES:
            # Dequeue from front (BFS uses FIFO)
            url, depth = self.to_visit.popleft()
            
            # Skip if already visited or depth exceeded
            if url in self.visited or depth > CRAWLER_MAX_DEPTH:
                continue
            
            # Mark as visited
            self.visited.add(url)
            
            # Fetch page
            response = fetch(url, self.timeout)
            if not response:
                continue
            
            # Extract page info
            title = self.extractor.extract_page_info(response.text)
            
            # Check if should crawl based on keywords
            if not self.should_crawl(url, title):
                print(f"[SKIP] Keywords not matched: {url}")
                continue
            
            # Download if corpus document
            if self.is_corpus_document(url):
                self.download_document(url, response)
            
            # Extract and queue new links for next depth level (BFS)
            if depth < CRAWLER_MAX_DEPTH:
                links = self.extractor.extract_links(url, response.text)
                
                for link in links:
                    if link not in self.visited and self.extractor.is_valid_url(link, start_url):
                        self.to_visit.append((link, depth + 1))
        
        if not self.results:
            print("\n[WARNING] No pages processed.")
            return []
        
        return self.results
    
