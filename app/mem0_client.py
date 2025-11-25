import os
from typing import Any, Dict, List, Optional

import requests


MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_USER_ID = os.getenv("MEM0_DEFAULT_USER_ID", None)
DEFAULT_AGENT_ID = os.getenv("MEM0_DEFAULT_AGENT_ID", None)


class Mem0Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_user_id: Optional[str] = DEFAULT_USER_ID,
        default_agent_id: Optional[str] = DEFAULT_AGENT_ID,
    ) -> None:
        self.base_url = base_url or MEM0_BASE_URL
        self.default_user_id = default_user_id
        self.default_agent_id = default_agent_id

    # ---- Core helpers ----

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _handle(self, resp: requests.Response) -> Any:
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    # ---- API methods ----

    def add_memories(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """POST /memories"""
        payload: Dict[str, Any] = {
            "messages": messages,
        }
        if user_id or self.default_user_id:
            payload["user_id"] = user_id or self.default_user_id
        if agent_id or self.default_agent_id:
            payload["agent_id"] = agent_id or self.default_agent_id
        if run_id:
            payload["run_id"] = run_id
        if metadata:
            payload["metadata"] = metadata

        resp = requests.post(self._url("/memories"), json=payload, timeout=30)
        return self._handle(resp)

    def list_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        """GET /memories"""
        params: Dict[str, Any] = {}
        if user_id or self.default_user_id:
            params["user_id"] = user_id or self.default_user_id
        if agent_id or self.default_agent_id:
            params["agent_id"] = agent_id or self.default_agent_id
        if run_id:
            params["run_id"] = run_id

        resp = requests.get(self._url("/memories"), params=params, timeout=30)
        return self._handle(resp)

    def get_memory(self, memory_id: str) -> Any:
        """GET /memories/{memory_id}"""
        resp = requests.get(self._url(f"/memories/{memory_id}"), timeout=30)
        return self._handle(resp)

    def update_memory(self, memory_id: str, data: Dict[str, Any]) -> Any:
        """PUT /memories/{memory_id}"""
        resp = requests.put(self._url(f"/memories/{memory_id}"), json=data, timeout=30)
        return self._handle(resp)

    def delete_memory(self, memory_id: str) -> Any:
        """DELETE /memories/{memory_id}"""
        resp = requests.delete(self._url(f"/memories/{memory_id}"), timeout=30)
        return self._handle(resp)

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        """DELETE /memories (all for a given identifier)"""
        params: Dict[str, Any] = {}
        if user_id or self.default_user_id:
            params["user_id"] = user_id or self.default_user_id
        if agent_id or self.default_agent_id:
            params["agent_id"] = agent_id or self.default_agent_id
        if run_id:
            params["run_id"] = run_id

        resp = requests.delete(self._url("/memories"), params=params, timeout=30)
        return self._handle(resp)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """POST /search"""
        payload: Dict[str, Any] = {"query": query}
        if user_id or self.default_user_id:
            payload["user_id"] = user_id or self.default_user_id
        if agent_id or self.default_agent_id:
            payload["agent_id"] = agent_id or self.default_agent_id
        if run_id:
            payload["run_id"] = run_id
        if filters:
            payload["filters"] = filters

        resp = requests.post(self._url("/search"), json=payload, timeout=60)
        return self._handle(resp)

    def reset(self) -> Any:
        """POST /reset"""
        resp = requests.post(self._url("/reset"), timeout=60)
        return self._handle(resp)