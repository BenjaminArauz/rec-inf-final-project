"""
Crawler utility functions and classes
Handles HTTP fetching and link extraction
"""
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


def fetch(url, timeout):
    """
    Fetch a page and return response or None on error.
    
    Parameters:
    - url: URL to fetch
    - timeout: Request timeout in seconds
    
    Returns:
    - Response object or None on error
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; RECINF-Crawler/1.0)'}
        )
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None


class LinkExtractor:
    """Extracts and validates links from HTML pages"""
    
    def __init__(self, domain_limit):
        self.domain_limit = domain_limit
    
    def is_valid_url(self, url, base_url):
        """Check if URL is valid and within domain limits"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_url)
            
            if parsed.scheme not in ('http', 'https'):
                return False
            
            if self.domain_limit and parsed.netloc != base_parsed.netloc:
                return False
            
            return True
        except Exception:
            return False
    
    def extract_links(self, page_url, html):
        """Extract and normalize all links from HTML"""
        links = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for a in soup.select('a[href]'):
                href = a.get('href', '')
                if not href:
                    continue
                
                # Resolve relative URLs
                abs_url = urljoin(page_url, href)
                
                # Normalize URL
                parsed = urlparse(abs_url)
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    normalized += '?' + parsed.query
                
                links.append(normalized)
        
        except Exception as e:
            print(f"[PARSE ERROR] {page_url}: {e}")
        
        return links
    
    def extract_page_info(self, html):
        """Extract title and link count from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else 'Sin título'
            return title
        except Exception as e:
            print(f"[INFO ERROR] parsing HTML: {e}")
            return "Error"
