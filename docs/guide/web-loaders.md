# Web Loaders

Prompt Amplifier provides powerful web-based loaders to ingest content from URLs, YouTube videos, sitemaps, and RSS feeds.

## WebLoader

Load and parse web pages to extract text content.

```python
from prompt_amplifier.loaders import WebLoader

# Basic usage
loader = WebLoader()
docs = loader.load("https://example.com/article")

# Load multiple URLs
docs = loader.load_batch([
    "https://example.com/page1",
    "https://example.com/page2",
])

# Custom extraction
loader = WebLoader(
    extract_links=True,      # Also extract links
    clean_html=True,         # Remove scripts/styles
    timeout=30,              # Request timeout
)
```

### How It Works

1. Fetches HTML content using `requests`
2. Parses with BeautifulSoup
3. Removes scripts, styles, navigation
4. Extracts clean text content
5. Preserves metadata (URL, title, description)

### Use Cases

- **Documentation ingestion**: Load product docs
- **News articles**: Fetch current events
- **Research papers**: Load web-published papers
- **Company info**: Scrape about pages

---

## YouTubeLoader

Extract transcripts from YouTube videos.

```python
from prompt_amplifier.loaders import YouTubeLoader

# Load video transcript
loader = YouTubeLoader()
docs = loader.load("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Or use video ID directly
docs = loader.load("dQw4w9WgXcQ")

# With language preference
loader = YouTubeLoader(languages=["en", "es"])
docs = loader.load("VIDEO_ID")
```

### Features

- Auto-extracts video ID from URL
- Supports multiple languages
- Falls back to auto-generated captions
- Preserves timestamps in metadata

### Use Cases

- **Tutorial content**: Load how-to videos
- **Lectures**: Ingest educational content
- **Podcasts**: Video podcasts with transcripts
- **Webinars**: Extract key information

---

## SitemapLoader

Crawl entire websites using their sitemap.

```python
from prompt_amplifier.loaders import SitemapLoader

# Load all pages from sitemap
loader = SitemapLoader()
docs = loader.load("https://example.com/sitemap.xml")

# Limit pages to crawl
loader = SitemapLoader(
    max_pages=50,            # Max pages to load
    filter_urls=["blog"],    # Only URLs containing "blog"
    exclude_urls=["archive"],# Skip archive pages
)
docs = loader.load("https://example.com/sitemap.xml")

# With concurrent loading
loader = SitemapLoader(
    max_pages=100,
    concurrent=True,
    max_workers=5,
)
```

### Features

- Parses XML sitemaps
- Supports sitemap index files
- URL filtering with patterns
- Concurrent page loading
- Respects robots.txt

### Use Cases

- **Full site ingestion**: Load entire documentation
- **Blog archives**: Index all blog posts
- **Product catalogs**: Scrape e-commerce sites
- **Knowledge bases**: Ingest support articles

---

## RSSLoader

Load content from RSS and Atom feeds.

```python
from prompt_amplifier.loaders import RSSLoader

# Load RSS feed
loader = RSSLoader()
docs = loader.load("https://example.com/feed.xml")

# Load multiple feeds
docs = loader.load_batch([
    "https://blog1.com/feed",
    "https://blog2.com/rss",
])

# With options
loader = RSSLoader(
    max_entries=20,          # Limit entries
    load_content=True,       # Fetch full content
)
```

### Features

- Supports RSS 2.0 and Atom feeds
- Extracts title, description, content
- Preserves publication dates
- Optional full content fetching

### Use Cases

- **News aggregation**: Multi-source news
- **Blog monitoring**: Track industry blogs
- **Podcast notes**: Load show descriptions
- **Research updates**: Academic feeds

---

## Complete Example

Build a knowledge base from multiple web sources:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.loaders import (
    WebLoader,
    YouTubeLoader,
    SitemapLoader,
    RSSLoader,
)

# Initialize PromptForge
forge = PromptForge()

# Load documentation
sitemap = SitemapLoader(max_pages=30)
forge.add_documents(sitemap.load("https://docs.example.com/sitemap.xml"))

# Load tutorials
youtube = YouTubeLoader()
video_ids = ["abc123", "def456", "ghi789"]
for vid in video_ids:
    forge.add_documents(youtube.load(vid))

# Load news
rss = RSSLoader(max_entries=10)
forge.add_documents(rss.load("https://news.example.com/feed"))

# Load specific pages
web = WebLoader()
forge.add_documents(web.load("https://example.com/faq"))

print(f"Total chunks: {forge.chunk_count}")

# Now expand prompts with this rich context
result = forge.expand("How do I get started?")
```

---

## Requirements

Install web loader dependencies:

```bash
pip install prompt-amplifier[loaders]
```

This installs:
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP client
- `youtube-transcript-api` - YouTube transcripts
- `feedparser` - RSS/Atom parsing

