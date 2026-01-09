import os
import requests
from bs4 import BeautifulSoup

def fetch_index(url: str):
    """
    Fetch the index HTML page from the given URL.
    Parameters:
    - url, URL of the index page
    Returns:
    - str: HTML content of the index page
    """

    resp = requests.get(url, timeout=20, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; RECINF-Crawler/1.0)'
    })
    resp.raise_for_status()
    return resp.text

def parse_corpus_links(index_html: str, base_url: str):
    """
    Parse the index HTML to extract corpus document links.
    Parameters:
    - index_html: str, HTML content of the index page
    - base_url: str, base URL to prepend to relative links
    Returns:
    - list of str: full URLs to corpus documents
    """

    soup = BeautifulSoup(index_html, 'html.parser')
    links = []
    for a in soup.select('a[href]'):
        href = a['href']
        if href.startswith('corpus/'):
            links.append(f"{base_url.rstrip('/')}/{href}")
    return links

def download_if_missing(doc_url: str, dest_dir: str):
    """
    Download the document from doc_url into dest_dir if not already present.
    Parameters:
    - doc_url: str, URL of the document to download
    - dest_dir: str, local directory to save the document
    Returns:
    - bool: True if downloaded, False if already exists
    """

    doc_id = doc_url.rstrip('/').split('/')[-1]
    dest_path = os.path.join(dest_dir, doc_id)

    if os.path.exists(dest_path):
        return False
    print(f"Download: {doc_url}")
    
    os.makedirs(dest_dir, exist_ok=True)
    resp = requests.get(doc_url, timeout=30, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; RECINF-Crawler/1.0)'
    })

    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(resp.content)
    return True

def sync_corpus_from_index(index_url: str, dest_dir: str):
    """
    Fetch index HTML, parse corpus links, download missing docs into dest_dir.
    Returns a summary dict with counts.
    """
    index_html = fetch_index(index_url)
    # Base URL is everything up to '/index.html'
    base_url = index_url.rsplit('/', 1)[0]
    links = parse_corpus_links(index_html, base_url)
    downloaded = 0
    skipped = 0
    for doc_url in links:
        try:
            changed = download_if_missing(doc_url, dest_dir)
            if changed:
                downloaded += 1
            else:
                skipped += 1
        except Exception:
            # Best-effort: skip on error
            skipped += 1
            continue
    return {"total": len(links), "downloaded": downloaded, "skipped": skipped}
