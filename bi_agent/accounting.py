"""Durable per-call reservations and usage; interrupted requests retain their allowance."""
from __future__ import annotations

import uuid
import json
from datetime import datetime, timezone

from .errors import StageError
from .llm import PRICES, PRICE_PER_SEARCH

TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def summary(records):
    totals = {k: sum((r.get("usage") or {}).get(k, 0) for r in records) for k in (*TOKEN_KEYS, "web_search_requests")}
    totals["calls"] = len(records)
    costs = [r.get("estimated_cost_usd") for r in records]
    totals["estimated_cost_usd"] = round(sum(costs), 6) if all(c is not None for c in costs) else None
    totals["unavailable_calls"] = sum(r["status"] != "completed" for r in records)
    return totals


class CallAccounting:
    def __init__(self, store, llm, stage):
        self.store, self.llm, self.stage = store, llm, stage

    def _load(self):
        return self.store.load_json("usage.json") if self.store.exists("usage.json") else {"records": [], "budget": {}}

    def _save(self, data):
        data["total"] = summary(data["records"])
        data["stages"] = {stage: summary([r for r in data["records"] if r["stage"] == stage])
                          for stage in {r["stage"] for r in data["records"]}}
        self.store.save_json("usage.json", data)
        self.store.checkpoint(only=["usage.json"])

    def reserve(self, label, kwargs):
        data = self._load()
        # Limits persist across CLI invocations; explicit CLI values revise them.
        for key, value in self.llm.run_budget.items():
            if value is not None:
                if value < 0:
                    raise StageError("run budgets must be nonnegative")
                data["budget"][key] = value
        searches = sum(t.get("max_uses", 0) for t in kwargs.get("tools", []) if t.get("name") == "web_search")
        # UTF-8 bytes upper-bound prompt tokens; the context allowance also covers server
        # tool results. Never reserve less than the serialized request already contains.
        input_allowance = max(self.llm.input_token_allowance, len(json.dumps(kwargs, ensure_ascii=False).encode()))
        tokens = input_allowance + self.llm.max_tokens
        price = PRICES.get(self.llm.model)
        cost = ((input_allowance * price[0] * 1.25 + self.llm.max_tokens * price[1]) / 1e6
                + searches * PRICE_PER_SEARCH) if price else None
        allowance = {"tokens": tokens, "searches": searches, "cost_usd": cost}
        for key, limit in data["budget"].items():
            used = 0
            for r in data["records"]:
                if r["status"] == "completed":
                    u = r["usage"]
                    charge = sum(u.get(k, 0) for k in TOKEN_KEYS) if key == "tokens" else (
                        u.get("web_search_requests", 0) if key == "searches" else r["estimated_cost_usd"])
                else:
                    charge = r["reservation"][key]
                if charge is None:
                    raise StageError("cost budget cannot admit calls with unknown pricing or usage")
                used += charge
            if allowance[key] is None or used + allowance[key] > limit:
                self._save(data)
                raise StageError(f"run budget exhausted ({key}); completed work is saved")
        record = {"call_id": uuid.uuid4().hex, "run_id": self.store.manifest()["run_id"],
                  "stage": self.stage, "attempt": self.store._attempt, "label": label,
                  "model": self.llm.model, "status": "interrupted", "usage": None,
                  "started_at": datetime.now(timezone.utc).isoformat(), "reservation": allowance,
                  "pricing": {"input_per_million": price[0] if price else None,
                              "output_per_million": price[1] if price else None,
                              "cache_write_multiplier": 1.25, "cache_read_multiplier": 0.1,
                              "search_usd": PRICE_PER_SEARCH, "basis": "configured estimate; not an invoice"},
                  "estimated_cost_usd": None}
        data["records"].append(record)
        self._save(data)
        return record["call_id"]

    def finish(self, call_id, usage=None, error=None):
        data = self._load()
        record = next(r for r in data["records"] if r["call_id"] == call_id)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        if usage is not None:
            record.update(status="completed", usage=usage.__dict__, estimated_cost_usd=usage.cost(record["model"]))
        else:
            record["error"] = error or "final usage unavailable"
        self._save(data)
