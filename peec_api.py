import os
import time
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class PeecAPIError(Exception):
    def __init__(self, status_code: int, response_body: Any):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"Peec API error {status_code}: {response_body}")


class PeecClient:
    BASE_URL = "https://api.peec.ai/customer/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        api_key = api_key or os.environ.get("PEEC_API_KEY")
        if not api_key:
            raise ValueError("API key required: pass api_key or set PEEC_API_KEY env var")
        self.session = requests.Session()
        self.session.headers["x-api-key"] = api_key

    def _request(self, method: str, path: str, params: Optional[dict] = None, json: Optional[dict] = None) -> Any:
        for attempt in range(5):
            resp = self.session.request(method, f"{self.base_url}{path}", params=params, json=json)
            if resp.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            if not resp.ok:
                raise PeecAPIError(resp.status_code, resp.text)
            return resp.json()
        raise PeecAPIError(429, "Rate limit exceeded after 5 retries")

    def _get(self, path: str, **params) -> Any:
        filtered = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=filtered or None)

    def _post(self, path: str, **kwargs) -> Any:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return self._request("POST", path, json=filtered or None)

    # ── GET endpoints ──

    def list_projects(self) -> list:
        return self._get("/projects")

    def list_brands(self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list:
        return self._get("/brands", project_id=project_id, limit=limit, offset=offset)

    def list_prompts(self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list:
        return self._get("/prompts", project_id=project_id, limit=limit, offset=offset)

    def list_tags(self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list:
        return self._get("/tags", project_id=project_id, limit=limit, offset=offset)

    def list_topics(self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list:
        return self._get("/topics", project_id=project_id, limit=limit, offset=offset)

    def list_models(self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list:
        return self._get("/models", project_id=project_id, limit=limit, offset=offset)

    def list_chats(
        self, project_id: str, limit: Optional[int] = None, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> list:
        return self._get("/chats", project_id=project_id, limit=limit, offset=offset,
                         start_date=start_date, end_date=end_date)

    def get_chat_content(self, chat_id: str, project_id: str) -> dict:
        return self._get(f"/chats/{chat_id}/content", project_id=project_id)

    # ── POST endpoints ──

    def report_brands(
        self, project_id: str, limit: int = 1000, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        dimensions: Optional[list] = None, filters: Optional[dict] = None,
    ) -> dict:
        return self._post("/reports/brands", project_id=project_id, limit=limit, offset=offset,
                          start_date=start_date, end_date=end_date, dimensions=dimensions, filters=filters)

    def report_domains(
        self, project_id: str, limit: int = 1000, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        dimensions: Optional[list] = None, filters: Optional[dict] = None,
    ) -> dict:
        return self._post("/reports/domains", project_id=project_id, limit=limit, offset=offset,
                          start_date=start_date, end_date=end_date, dimensions=dimensions, filters=filters)

    def report_urls(
        self, project_id: str, limit: int = 1000, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        dimensions: Optional[list] = None, filters: Optional[dict] = None,
    ) -> dict:
        return self._post("/reports/urls", project_id=project_id, limit=limit, offset=offset,
                          start_date=start_date, end_date=end_date, dimensions=dimensions, filters=filters)

    def query_search(
        self, project_id: str, limit: int = 1000, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        dimensions: Optional[list] = None, filters: Optional[dict] = None,
    ) -> dict:
        return self._post("/queries/search", project_id=project_id, limit=limit, offset=offset,
                          start_date=start_date, end_date=end_date, dimensions=dimensions, filters=filters)

    def query_shopping(
        self, project_id: str, limit: int = 1000, offset: Optional[int] = None,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        dimensions: Optional[list] = None, filters: Optional[dict] = None,
    ) -> dict:
        return self._post("/queries/shopping", project_id=project_id, limit=limit, offset=offset,
                          start_date=start_date, end_date=end_date, dimensions=dimensions, filters=filters)

    # ── Pagination helper ──

    def fetch_all(self, method: Callable, page_size: int = 1000, **kwargs) -> list:
        all_results = []
        offset = 0
        while True:
            page = method(limit=page_size, offset=offset, **kwargs)
            if isinstance(page, list):
                items = page
            elif isinstance(page, dict) and "data" in page:
                items = page["data"]
            else:
                all_results.append(page)
                break
            all_results.extend(items)
            if len(items) < page_size:
                break
            offset += page_size
        return all_results


if __name__ == "__main__":
    client = PeecClient()
    projects = client.list_projects()
    print(f"Projects: {projects}")

    if projects.get("data"):
        pid = projects["data"][0]["id"]
        brands = client.list_brands(pid)
        print(f"Brands: {brands}")

        report = client.report_brands(pid, start_date="2026-01-01", end_date="2026-03-18")
        print(f"Brand report: {report}")
