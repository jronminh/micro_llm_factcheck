"""Search DuckDuckGo HTML endpoint, không cần API key."""
import sys

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; micro-llm-factcheck/0.1)"}
MAX_PAGES = 6  # ~10-15 link/trang -> đủ cho max_n tới 50-60, chặn vòng lặp vô hạn


def _parse_results(html: str) -> tuple[list[dict], dict | None]:
    """Trả về (results của trang này, form-data để lấy trang kế - None nếu hết trang)."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for result in soup.select(".result"):
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")
        if not title_el:
            continue
        results.append({
            "title": title_el.get_text(separator=" ", strip=True),
            "snippet": snippet_el.get_text(separator=" ", strip=True) if snippet_el else "",
            "url": title_el.get("href", ""),
        })

    # Trang kế được DDG trả qua 1 <form> ẩn chứa offset "s" + token "vqd" của
    # phiên tìm kiếm hiện tại - không tự bịa offset, phải lấy đúng field DDG
    # cung cấp (bao gồm vqd, đổi mỗi phiên) thì mới hợp lệ.
    next_inputs = None
    for form in soup.select("form"):
        inputs = {i.get("name"): i.get("value") for i in form.select("input") if i.get("name")}
        if inputs.get("s") and inputs.get("vqd"):
            next_inputs = inputs
            break
    return results, next_inputs


def search(query: str, n: int = 5) -> list[dict]:
    session = requests.Session()
    all_results: list[dict] = []
    seen_urls: set[str] = set()
    payload = {"q": query}

    for _ in range(MAX_PAGES):
        resp = session.post(SEARCH_URL, data=payload, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        page_results, next_inputs = _parse_results(resp.text)

        for r in page_results:
            if r["url"] and r["url"] in seen_urls:
                continue  # trang sau có thể lặp lại vài kết quả của trang trước
            seen_urls.add(r["url"])
            all_results.append(r)

        if len(all_results) >= n or not next_inputs:
            break
        payload = {**next_inputs, "q": query}

    return all_results[:n]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "python programming language"
    for r in search(q):
        print(f"- {r['title']}\n  {r['snippet']}\n  {r['url']}\n")
