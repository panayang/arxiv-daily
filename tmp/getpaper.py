import sqlite3
import httpx
import feedparser
import json
import re
import toml
from datetime import datetime

# Configuration
DB_NAME = "arxiv_dailypy.db"
CONFIG_FILE = "config.toml"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return toml.load(f)

def get_arxiv_url(config):
    cat = config['arxiv']['category']
    start = config['arxiv']['start']
    max_results = config['arxiv']['max_results']
    return f"https://export.arxiv.org/api/query?search_query=cat:{cat}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                updated DATETIME,
                published DATETIME,
                summary TEXT,
                primary_category TEXT,
                categories TEXT,
                authors TEXT,
                pdf_link TEXT
            )
        """)
        conn.commit()

def clean_id(raw_id: str) -> str:
    """
    Mirroring the Rust logic: Extracts the ID after '/abs/' 
    and strips the version suffix (e.g., 'v1').
    """
    if "/abs/" in raw_id:
        # Get everything after /abs/
        s = raw_id.split("/abs/")[-1]
        # Remove version suffix like v1, v2 if it's numeric
        return re.sub(r'v\d+$', '', s)
    return raw_id

def transform_entry(entry):
    authors = [a.name for a in entry.get('authors', [])]
    
    # Extract PDF link specifically
    pdf_link = None
    for link in entry.get('links', []):
        if link.get('type') == 'application/pdf':
            pdf_link = link.get('href')
            break
    
    all_categories = [c.term for c in entry.get('tags', [])]
    primary_cat = all_categories[0] if all_categories else ""

    return {
        "id": clean_id(entry.id),
        "url": entry.id,
        "title": entry.title.replace('\n', ' ').strip() if 'title' in entry else "",
        "updated": entry.updated,
        "published": entry.published,
        "summary": entry.summary if 'summary' in entry else "",
        "primary_category": primary_cat,
        "categories": ",".join(all_categories),
        "authors": json.dumps(authors),
        "pdf_link": pdf_link
    }

def main():
    init_db()
    config = load_config()
    arxiv_url = get_arxiv_url(config)
    
    print(f"Fetching from arXiv (cat:{config['arxiv']['category']}, start:{config['arxiv']['start']}, max:{config['arxiv']['max_results']})...")
    # follow_redirects=True is the key fix for the 301 error
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(arxiv_url, timeout=60.0)
            response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    print("Parsing feed...")
    feed = feedparser.parse(response.content)
    print(f"Found {len(feed.entries)} entries. Processing...")

    papers = [transform_entry(e) for e in feed.entries]

    if not papers:
        print("No papers found in the feed.")
        return

    print(f"Saving {len(papers)} papers to database...")
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        query = """
            INSERT INTO papers (id, url, title, updated, published, summary, primary_category, categories, authors, pdf_link)
            VALUES (:id, :url, :title, :updated, :published, :summary, :primary_category, :categories, :authors, :pdf_link)
            ON CONFLICT(id) DO UPDATE SET
                updated = excluded.updated,
                title = excluded.title,
                summary = excluded.summary,
                url = excluded.url
        """
        
        cursor.executemany(query, papers)
        conn.commit()

    print(f"Success! {len(papers)} papers saved to {DB_NAME}.")

if __name__ == "__main__":
    main()