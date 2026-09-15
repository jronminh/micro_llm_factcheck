"""Search DuckDuckGo HTML endpoint, không cần API key."""
import sys

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; micro-llm-factcheck/0.1)"}


def search(query: str, n: int = 5) -> list[dict]:
    resp = requests.post(SEARCH_URL, data={"q": query}, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for result in soup.select(".result")[:n]:
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")
        if not title_el:
            continue
        results.append({
            "title": title_el.get_text(separator=" ", strip=True),
            "snippet": snippet_el.get_text(separator=" ", strip=True) if snippet_el else "",
            "url": title_el.get("href", ""),
        })
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "python programming language"
    for r in search(q):
        print(f"- {r['title']}\n  {r['snippet']}\n  {r['url']}\n")
