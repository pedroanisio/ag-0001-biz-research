"""Atomic content-addressed stage snapshots with provenance and a process writer lock.

manifest.json is the commit point. Top-level files are convenient projections; readers
use immutable objects referenced by the manifest, even after a crash during publication.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from .errors import StageError
from .i18n import DEFAULT_LANG, SUPPORTED, normalize_lang
from .models import Evidence, EvidenceLedger

SCHEMA_VERSION = 2
INPUTS = {
    "crawl": [],
    "identify": ["run.json", "pages.json", "evidence.crawl.json"],
    "signals": ["run.json", "pages.json", "evidence.crawl.json", "identity.site.json"],
    "research": ["run.json", "pages.json", "evidence.crawl.json", "identity.site.json", "signals.json"],
    "resolve": ["run.json", "evidence.json", "identity.site.json", "findings.json"],
    "analyze": ["run.json", "evidence.json", "identity.json", "signals.json", "findings.json"],
    "narrate": ["run.json", "evidence.json", "identity.json", "signals.json", "findings.json", "analysis.json"],
    "report": ["run.json", "evidence.json", "identity.json", "signals.json", "findings.json", "analysis.json", "claims.json", "narrative.json"],
}
OUTPUTS = {"crawl": "pages.json", "identify": "identity.site.json", "signals": "signals.json",
           "research": "findings.json", "resolve": "identity.json", "analyze": "analysis.json",
           "narrate": "narrative.json", "report": "report.pdf"}


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        d = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def encoded(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode()


class RunStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._locked = False
        self._pending: dict[str, bytes] = {}
        self._stage = None
        self._config = {}
        self._inputs = {}
        self._attempt = 0

    def path(self, name):
        return self.dir / name

    def manifest(self):
        path = self.path("manifest.json")
        if not path.exists():
            return {"schema_version": SCHEMA_VERSION, "run_id": None, "artifacts": {}, "attempts": {}}
        data = json.loads(path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise StageError("incompatible run schema; crawl into a new run revision")
        return data

    @contextmanager
    def writer(self):
        if self._locked:
            yield
            return
        with self.path(".writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StageError("another process is writing this run") from exc
            self._locked = True
            try:
                yield
            finally:
                self._locked = False
                fcntl.flock(lock, fcntl.LOCK_UN)

    def exists(self, name):
        return name in self._pending or name in self.manifest()["artifacts"]

    def _fresh(self, name, manifest, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            return False
        seen = seen | {name}
        entry = manifest["artifacts"].get(name)
        if entry is None or entry.get("run_id") != manifest["run_id"]:
            return False
        for dep, digest in entry.get("inputs", {}).items():
            other = manifest["artifacts"].get(dep)
            if not other or other["sha256"] != digest or not self._fresh(dep, manifest, seen):
                return False
        return True

    def read_bytes(self, name):
        if name in self._pending:
            return self._pending[name]
        manifest = self.manifest()
        if name not in manifest["artifacts"]:
            if self.path(name).exists():
                raise StageError(f"unversioned {name}; re-run crawl and dependent stages (legacy artifacts are not trusted)")
            raise StageError(f"missing {name}; run the earlier stage first")
        if not self._fresh(name, manifest):
            raise StageError(f"stale {name}; re-run its producing stage")
        entry = manifest["artifacts"][name]
        try:
            data = self.path(entry["object"]).read_bytes()
        except OSError as exc:
            raise StageError(f"incomplete artifact {name}") from exc
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise StageError(f"corrupt artifact {name}")
        return data

    def load_json(self, name):
        return json.loads(self.read_bytes(name))

    def load_model(self, name, schema):
        return schema.model_validate(self.load_json(name))

    def save_bytes(self, name, data):
        self._pending[name] = data
        if self._stage is None:
            self.checkpoint()

    def save_json(self, name, data):
        self.save_bytes(name, encoded(data))

    def save_model(self, name, model):
        self.save_json(name, model.model_dump(mode="json"))

    def ledger(self):
        name = "evidence.crawl.json" if self._stage in {"identify", "signals"} else "evidence.json"
        if name not in self._pending and not self._fresh(name, self.manifest()) and self.exists("evidence.crawl.json"):
            name = "evidence.crawl.json"
        return EvidenceLedger([Evidence.model_validate(e) for e in self.load_json(name)])

    def save_ledger(self, ledger):
        self.save_json("evidence.json", [e.model_dump(mode="json") for e in ledger])

    def meta(self):
        return self.load_json("run.json")

    def site_lang(self):
        return self.meta().get("site_lang", DEFAULT_LANG)

    def lang(self):
        return self.meta().get("lang") or self.site_lang()

    def set_lang(self, lang):
        code = normalize_lang(lang)
        if code is None:
            raise StageError(f"unsupported language {lang!r}; choose from {', '.join(SUPPORTED)}")
        with self.writer():
            meta = self.meta()
            meta["lang"] = code
            self.save_json("run.json", meta)

    def checkpoint(self, *, only=None, new_run=False):
        with self.writer():
            manifest = self.manifest()
            previous_artifacts = set(manifest["artifacts"])
            if new_run:
                manifest["run_id"] = uuid.uuid4().hex
                # Old immutable objects remain inspectable, but no artifact can cross run IDs.
                manifest["artifacts"] = {}
                manifest["attempts"] = {}
            if not manifest["run_id"]:
                manifest["run_id"] = uuid.uuid4().hex
            names = list(self._pending) if only is None else [n for n in only if n in self._pending]
            for name in names:
                data = self._pending[name]
                digest = hashlib.sha256(data).hexdigest()
                obj = f"objects/{digest}"
                if not self.path(obj).exists():
                    atomic_write(self.path(obj), data)
                independent = name in {"usage.json", "audit.json"}
                manifest["artifacts"][name] = {
                    "object": obj, "sha256": digest, "run_id": manifest["run_id"],
                    "schema_version": SCHEMA_VERSION, "stage": self._stage,
                    "configuration": self._config if not independent else {},
                    "inputs": self._inputs if not independent and self._stage != "crawl" else {},
                }
            if self._stage:
                manifest["attempts"][self._stage] = self._attempt
            manifest["commit_id"] = uuid.uuid4().hex
            atomic_write(self.path(f"commits/{manifest['commit_id']}.json"), encoded(manifest))
            atomic_write(self.path("manifest.json"), encoded(manifest))
            for name in names:
                # Projections can be recreated from the manifest; never read as stage inputs.
                atomic_write(self.path(name), self._pending.pop(name))
            if new_run:
                for obsolete in previous_artifacts - set(manifest["artifacts"]):
                    self.path(obsolete).unlink(missing_ok=True)

    @contextmanager
    def stage(self, name, config):
        with self.writer():
            self._stage, self._config = name, config
            manifest = self.manifest()
            self._attempt = manifest["attempts"].get(name, 0) + 1
            self._inputs = {}
            try:
                for dep in INPUTS[name]:
                    self.read_bytes(dep)
                    self._inputs[dep] = manifest["artifacts"][dep]["sha256"]
                yield
                self.checkpoint(new_run=name == "crawl")
            finally:
                self._pending.clear()
                self._stage = None
                self._config = {}
                self._inputs = {}

    def status(self):
        m = self.manifest()
        result = {}
        for stage, out in OUTPUTS.items():
            if out in m["artifacts"]:
                entry = m["artifacts"][out]
                complete = self._fresh(out, m) and (stage != "resolve" or entry["stage"] == "resolve")
                if not self.path(entry["object"]).exists():
                    result[stage] = "incomplete"
                else:
                    result[stage] = "completed" if complete else "stale"
            else:
                result[stage] = "runnable" if all(self._fresh(x, m) for x in INPUTS[stage]) else "incomplete"
        if self.exists("research.partial.json") and self._fresh("research.partial.json", m):
            p = self.load_json("research.partial.json")
            if any(g["status"] != "completed" for g in p.get("groups", {}).values()):
                result["research"] = "incomplete"
        return result


def staged(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(store, *args, **kwargs):
            from . import prompts
            llm = next((x for x in args if hasattr(x, "model")), kwargs.get("llm"))
            config = {"model": getattr(llm, "model", None),
                      "prompt_sha256": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
                      "options": {}}
            if name == "crawl":
                crawler = next((x for x in args if hasattr(x, "max_pages")), kwargs.get("crawler"))
                if crawler:
                    config["crawler"] = {k: getattr(crawler, k) for k in
                                         ("max_pages", "max_fetches", "delay", "allow_private", "max_bytes")}
            if llm:
                config.update({k: getattr(llm, k) for k in ("max_tokens", "max_search_uses", "max_fetch_uses", "max_attempts", "max_research_turns", "max_fetch_tokens", "thinking")})
            with store.stage(name, config):
                return fn(store, *args, **kwargs)
        return wrapped
    return decorate
