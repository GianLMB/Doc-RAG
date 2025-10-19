import logging
import time
from io import BytesIO
from urllib.parse import urljoin, urlparse

import pypdf
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from .utils import setup_logger


class DocumentationScraper:
    def __init__(
        self, base_url: str, max_pages: int = 100, log_level: int = logging.INFO
    ):
        self.base_url = base_url.rstrip("/")  # Normalize base URL
        self.base_path = urlparse(base_url).path.rstrip("/")  # Get base path
        self.max_pages = max_pages
        self.visited_urls: set[str] = set()
        self.queued_urls: set[str] = set()  # Track already queued URLs
        self.documents: list[dict[str, str]] = []
        self.domain = urlparse(base_url).netloc
        self.logger = setup_logger(self, level=log_level)

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain and base path, and hasn't been seen."""
        parsed = urlparse(url)

        # Check domain matches
        if parsed.netloc != self.domain:
            return False

        # Check if URL starts with the base path
        url_path = parsed.path.rstrip("/")
        if not url_path.startswith(self.base_path):
            return False

        # Check if already visited or queued
        if url in self.visited_urls or url in self.queued_urls:
            return False

        # Check file extensions
        if url.endswith((".zip", ".tar.gz", ".jpg", ".png", ".gif", ".svg", ".ico")):  # noqa: SIM103
            return False

        return True

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        # Remove fragment
        url = url.split("#")[0]
        # Remove trailing slash for consistency (except for root)
        parsed = urlparse(url)
        if parsed.path != "/":
            url = url.rstrip("/")
        return url

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """Extract all valid links from the page."""
        links = []
        for link in soup.find_all("a", href=True):
            absolute_url = urljoin(current_url, link["href"])
            absolute_url = self.normalize_url(absolute_url)

            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
                self.queued_urls.add(absolute_url)  # Mark as queued

        return links

    def extract_content(self, soup: BeautifulSoup, url: str) -> dict[str, str]:
        """Extract meaningful content from the page."""
        # Try to find main content area (common patterns)
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=["content", "documentation", "docs"])
            or soup.find("body")
        )

        # Remove script, style, and navigation elements
        for element in main_content.find_all(
            ["script", "style", "nav", "header", "footer"]
        ):
            element.decompose()

        # Get title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else url

        # Convert to markdown
        content_html = str(main_content)
        content_md = md(content_html, heading_style="ATX")

        return {
            "url": url,
            "title": title_text,
            "content": content_md,
        }

    def extract_pdf_content(self, pdf_bytes: bytes, url: str) -> dict[str, str]:
        """Extract content from PDF file."""
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = pypdf.PdfReader(pdf_file)

            # Extract text from all pages
            text_content = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)

            content = "\n\n".join(text_content)

            # Try to get title from PDF metadata or first line
            title = url.split("/")[-1]  # Default to filename
            if pdf_reader.metadata and pdf_reader.metadata.title:
                title = pdf_reader.metadata.title
            elif content:
                # Use first non-empty line as title
                first_line = content.split("\n")[0].strip()
                if first_line and len(first_line) < 200:
                    title = first_line

            return {
                "url": url,
                "title": title,
                "content": content,
            }
        except Exception as e:
            self.logger.warning(f"\nError parsing PDF {url}: {e}")
            return {
                "url": url,
                "title": url.split("/")[-1],
                "content": "",
            }

    def scrape(self) -> list[dict[str, str]]:
        """Scrape documentation starting from base URL."""
        # Normalize and add base URL
        normalized_base = self.normalize_url(self.base_url)
        urls_to_visit = [normalized_base]
        self.queued_urls.add(normalized_base)
        total_pages = min(self.max_pages, len(urls_to_visit))

        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            url = urls_to_visit.pop(0)
            self.logger.info(
                f"Scraping ({len(self.visited_urls) + 1}/{total_pages}): {url}"
            )

            if url in self.visited_urls:
                continue

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                self.visited_urls.add(url)

                # Check if it's a PDF
                content_type = response.headers.get("content-type", "").lower()
                is_pdf = url.endswith(".pdf") or "application/pdf" in content_type

                if is_pdf:
                    # Handle PDF
                    doc = self.extract_pdf_content(response.content, url)
                    if doc["content"].strip():
                        self.documents.append(doc)
                else:
                    # Handle HTML
                    soup = BeautifulSoup(response.content, "html.parser")

                    # Extract content
                    doc = self.extract_content(soup, url)
                    if doc["content"].strip():
                        self.documents.append(doc)

                    # Extract and add new links
                    new_links = self.extract_links(soup, url)
                    urls_to_visit.extend(new_links)

                total_pages = min(self.max_pages, len(urls_to_visit))

                # Do not overload the server
                time.sleep(0.51)

            except Exception as e:
                self.logger.warning(f"\nError scraping {url}: {e}")
                continue

        return self.documents
