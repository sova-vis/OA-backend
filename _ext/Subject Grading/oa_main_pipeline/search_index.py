"""Local embedding search index for fast question retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .question_matcher import MatchResult, matcher_text_for_subject, rerank_search_results, tokenize
from .schemas import DataSourceLabel, QuestionRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchIndexArtifacts:
    source: DataSourceLabel
    root_dir: Path
    manifest_path: Path
    records_path: Path
    embeddings_path: Path


@dataclass(frozen=True)
class SearchIndexData:
    source: DataSourceLabel
    manifest: Dict[str, Any]
    record_ids: List[str]
    embeddings: np.ndarray
    positions_by_id: Dict[str, int]


@dataclass(frozen=True)
class SearchResult:
    match_result: MatchResult
    debug: Dict[str, Any]


class _HashingEmbedder:
    """Deterministic local fallback when sentence-transformers is unavailable."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.backend_name = "hash_fallback"
        self.resolved_model_name = f"hash-token-{dimension}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = tokenize(text) or [str(text or "").strip().casefold() or "empty"]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if (digest[4] % 2) == 0 else -1.0
                matrix[row_index, slot] += sign
            norm = float(np.linalg.norm(matrix[row_index]))
            if norm > 0:
                matrix[row_index] /= norm
        return matrix


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:  # pragma: no cover - dependency is optional in tests
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - fallback covers missing dependency
            raise RuntimeError("sentence-transformers is unavailable") from exc
        self._model = SentenceTransformer(model_name)
        self.backend_name = "sentence_transformers"
        self.resolved_model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class SearchIndexManager:
    """Builds, caches, and queries per-source local embedding indexes."""

    def __init__(
        self,
        *,
        repository: object,
        fallback_repository: object,
        main_repository: Optional[object] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.repository = repository
        self.fallback_repository = fallback_repository
        self.main_repository = main_repository
        self._embedder: Optional[object] = None
        self._embedder_lock = threading.Lock()
        self._source_locks: Dict[DataSourceLabel, threading.Lock] = {
            "o_level_json": threading.Lock(),
            "oa_main_dataset": threading.Lock(),
            "o_level_main_json": threading.Lock(),
        }
        self._loaded: Dict[DataSourceLabel, SearchIndexData] = {}

    def reload(self) -> None:
        self._loaded = {}

    def ensure_built(self, source: DataSourceLabel) -> Dict[str, Any]:
        artifacts = self._artifacts_for(source)
        lock = self._source_locks[source]
        rebuilt = False
        rebuild_reason: Optional[str] = None
        with lock:
            should_rebuild, rebuild_reason = self._should_rebuild(source, artifacts)
            if should_rebuild:
                started = time.perf_counter()
                logger.info(
                    "search_index build starting source=%s reason=%s cache_dir=%s",
                    source,
                    rebuild_reason,
                    str(artifacts.root_dir),
                )
                self._build_index(source, artifacts)
                logger.info(
                    "search_index build done source=%s elapsed_ms=%d",
                    source,
                    int((time.perf_counter() - started) * 1000),
                )
                self._loaded.pop(source, None)
                rebuilt = True
            if source not in self._loaded:
                logger.info("search_index load starting source=%s", source)
                started_load = time.perf_counter()
                self._loaded[source] = self._load_index(source, artifacts)
                logger.info(
                    "search_index load done source=%s elapsed_ms=%d records=%d",
                    source,
                    int((time.perf_counter() - started_load) * 1000),
                    len(self._loaded[source].record_ids),
                )
        return {
            "index_rebuilt": rebuilt,
            "rebuild_reason": rebuild_reason,
            "index_version": self._loaded[source].manifest.get("index_version"),
            "embedding_backend": self._loaded[source].manifest.get("embedding_backend"),
            "embedding_model": self._loaded[source].manifest.get("embed_model_resolved"),
        }

    def search(
        self,
        *,
        source: DataSourceLabel,
        query: str,
        query_subject: Optional[str] = None,
        student_answer: str,
        records: Sequence[QuestionRecord],
        use_answer_hint: bool = True,
    ) -> SearchResult:
        search_started = time.perf_counter()
        build_state = self.ensure_built(source)
        data = self._loaded[source]

        scoped_map = {record.question_id: record for record in records}
        scoped_positions = [
            data.positions_by_id[record.question_id]
            for record in records
            if record.question_id in data.positions_by_id
        ]
        if not scoped_positions:
            return SearchResult(
                match_result=MatchResult(
                    status="failed",
                    match_confidence=0.0,
                    best_record=None,
                    top_alternatives=[],
                ),
                debug={
                    "search_method": self.config.search_method,
                    "index_source": source,
                    "use_answer_hint": bool(use_answer_hint),
                    "index_version": build_state.get("index_version"),
                    "index_rebuilt": build_state.get("index_rebuilt", False),
                    "records_scoped": 0,
                    "top_k_searched": 0,
                    "query_embedding_ms": 0,
                    "similarity_search_ms": 0,
                    "rerank_ms": 0,
                    "final_search_score": 0.0,
                    "top_search_candidates": [],
                    "embedding_backend": build_state.get("embedding_backend"),
                    "embedding_model": build_state.get("embedding_model"),
                    "search_total_ms": int((time.perf_counter() - search_started) * 1000),
                },
            )

        embed_started = time.perf_counter()
        normalized_query = matcher_text_for_subject(query, subject=query_subject)
        query_embedding = self._get_embedder().encode([normalized_query])[0]
        query_embedding_ms = int((time.perf_counter() - embed_started) * 1000)

        similarity_started = time.perf_counter()
        scoped_embeddings = data.embeddings[scoped_positions]
        similarities = np.dot(scoped_embeddings, query_embedding).astype(np.float32)
        normalized_scores = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
        top_k = min(self.config.search_top_k, len(scoped_positions))
        if top_k < len(scoped_positions):
            top_local = np.argpartition(normalized_scores, -top_k)[-top_k:]
            top_local = top_local[np.argsort(normalized_scores[top_local])[::-1]]
        else:
            top_local = np.argsort(normalized_scores)[::-1]
        similarity_search_ms = int((time.perf_counter() - similarity_started) * 1000)

        candidate_records: List[QuestionRecord] = []
        embedding_scores: Dict[str, float] = {}
        for local_index in top_local.tolist():
            position = scoped_positions[local_index]
            record_id = data.record_ids[position]
            record = scoped_map.get(record_id)
            if record is None:
                continue
            candidate_records.append(record)
            embedding_scores[record.question_id] = float(normalized_scores[local_index])

        rerank_started = time.perf_counter()
        rerank_records = candidate_records[: self.config.search_rerank_k]
        match_result, candidate_debug = rerank_search_results(
            query,
            student_answer,
            rerank_records,
            embedding_scores,
            use_answer_hint=use_answer_hint,
            query_subject=query_subject,
            config=self.config,
        )
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)

        return SearchResult(
            match_result=match_result,
            debug={
                "search_method": self.config.search_method,
                "index_source": source,
                "use_answer_hint": bool(use_answer_hint),
                "query_subject": query_subject,
                "normalized_query_text": normalized_query,
                "index_version": build_state.get("index_version"),
                "index_rebuilt": build_state.get("index_rebuilt", False),
                "rebuild_reason": build_state.get("rebuild_reason"),
                "records_scoped": len(scoped_positions),
                "top_k_searched": top_k,
                "query_embedding_ms": query_embedding_ms,
                "similarity_search_ms": similarity_search_ms,
                "rerank_ms": rerank_ms,
                "final_search_score": match_result.match_confidence,
                "top_search_candidates": candidate_debug[: self.config.top_alternatives],
                "embedding_backend": build_state.get("embedding_backend"),
                "embedding_model": build_state.get("embedding_model"),
                "search_total_ms": int((time.perf_counter() - search_started) * 1000),
            },
        )

    def _artifacts_for(self, source: DataSourceLabel) -> SearchIndexArtifacts:
        root_dir = self.config.search_cache_dir / source
        return SearchIndexArtifacts(
            source=source,
            root_dir=root_dir,
            manifest_path=root_dir / "manifest.json",
            records_path=root_dir / "records.jsonl",
            embeddings_path=root_dir / "embeddings.npy",
        )

    def _repo_for_source(self, source: DataSourceLabel) -> object:
        if source == "o_level_json":
            return self.fallback_repository
        if source == "o_level_main_json":
            return self.main_repository
        return self.repository

    def _load_index(self, source: DataSourceLabel, artifacts: SearchIndexArtifacts) -> SearchIndexData:
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(artifacts.embeddings_path, allow_pickle=False)
        record_ids: List[str] = []
        for raw in artifacts.records_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            record_ids.append(str(payload.get("question_id") or ""))
        positions_by_id = {record_id: index for index, record_id in enumerate(record_ids) if record_id}
        return SearchIndexData(
            source=source,
            manifest=manifest,
            record_ids=record_ids,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            positions_by_id=positions_by_id,
        )

    def _should_rebuild(
        self,
        source: DataSourceLabel,
        artifacts: SearchIndexArtifacts,
    ) -> Tuple[bool, str]:
        if not artifacts.manifest_path.exists():
            return True, "missing_manifest"
        if not artifacts.records_path.exists():
            return True, "missing_records"
        if not artifacts.embeddings_path.exists():
            return True, "missing_embeddings"
        if not self.config.search_auto_rebuild:
            return False, "auto_rebuild_disabled"

        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        source_files = self._source_files_for(source)
        current_signature = self._source_signature(source_files)
        current_records = self._repo_for_source(source).get_records()  # type: ignore[attr-defined]
        current_record_count = len(current_records)
        current_record_signature = self._records_signature(current_records)
        embedder = self._get_embedder()
        if manifest.get("source_signature") != current_signature:
            return True, "source_signature_changed"
        if int(manifest.get("record_count") or 0) != current_record_count:
            return True, "record_count_changed"
        if str(manifest.get("record_signature") or "") != current_record_signature:
            return True, "record_signature_changed"
        if str(manifest.get("source_priority") or "") != self.config.source_priority:
            return True, "source_priority_changed"
        if str(manifest.get("embed_model_requested") or "") != self.config.embed_model:
            return True, "embed_model_requested_changed"
        if str(manifest.get("embed_model_resolved") or "") != getattr(embedder, "resolved_model_name", ""):
            return True, "embed_model_resolved_changed"
        if str(manifest.get("embedding_backend") or "") != getattr(embedder, "backend_name", ""):
            return True, "embedding_backend_changed"
        if int(manifest.get("index_version") or 0) != 2:
            return True, "index_version_changed"
        return False, "up_to_date"

    def _build_index(self, source: DataSourceLabel, artifacts: SearchIndexArtifacts) -> None:
        repo = self._repo_for_source(source)
        records = repo.get_records()  # type: ignore[attr-defined]
        texts = [matcher_text_for_subject(record.question_text, subject=record.subject) for record in records]
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts) if texts else np.zeros((0, 384), dtype=np.float32)
        source_files = self._source_files_for(source)
        artifacts.root_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "index_version": 2,
            "source": source,
            "built_at": int(time.time()),
            "source_priority": self.config.source_priority,
            "record_count": len(records),
            "record_signature": self._records_signature(records),
            "source_file_count": len(source_files),
            "source_signature": self._source_signature(source_files),
            "embed_model_requested": self.config.embed_model,
            "embed_model_resolved": getattr(embedder, "resolved_model_name", self.config.embed_model),
            "embedding_backend": getattr(embedder, "backend_name", "unknown"),
        }

        records_lines = [
            json.dumps(asdict(record), ensure_ascii=True)
            for record in records
        ]
        records_tmp = artifacts.records_path.with_suffix(".jsonl.tmp")
        manifest_tmp = artifacts.manifest_path.with_suffix(".json.tmp")
        embeddings_tmp = artifacts.embeddings_path.with_suffix(".npy.tmp")

        records_tmp.write_text("\n".join(records_lines), encoding="utf-8")
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
        with embeddings_tmp.open("wb") as handle:
            np.save(handle, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)

        records_tmp.replace(artifacts.records_path)
        manifest_tmp.replace(artifacts.manifest_path)
        embeddings_tmp.replace(artifacts.embeddings_path)

    def _source_files_for(self, source: DataSourceLabel) -> List[Path]:
        if source == "o_level_main_json":
            root = self.config.main_json_root
            return sorted(root.rglob("*.json")) if root.exists() else []
        repo = self._repo_for_source(source)
        if not hasattr(repo, "_load_records"):
            return []
        if source == "o_level_json":
            root = self.config.fallback_root
            return sorted(root.rglob("*.json")) if root.exists() else []

        root = self.config.dataset_root
        if not root.exists():
            return []
        out: List[Path] = []
        for summary_path in sorted(root.rglob(self.config.pair_summary_filename)):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(summary.get("status") or "").strip().lower() != "accepted":
                continue
            out.append(summary_path)
            qp_path = summary_path.parent / self.config.qp_extracted_filename
            ms_path = summary_path.parent / self.config.ms_extracted_filename
            if qp_path.exists():
                out.append(qp_path)
            if ms_path.exists():
                out.append(ms_path)
        return out

    def _source_signature(self, source_files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for path in source_files:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        return digest.hexdigest()

    def _records_signature(self, records: Sequence[QuestionRecord]) -> str:
        digest = hashlib.sha256()
        for record in records:
            digest.update(record.question_id.encode("utf-8"))
            digest.update(record.question_text.encode("utf-8"))
            digest.update(record.marking_scheme_answer.encode("utf-8"))
            digest.update(str(record.subject).encode("utf-8"))
            digest.update(str(record.year).encode("utf-8"))
            digest.update(str(record.session).encode("utf-8"))
            digest.update(str(record.paper).encode("utf-8"))
            digest.update(str(record.variant).encode("utf-8"))
        return digest.hexdigest()

    def _get_embedder(self) -> object:
        if self._embedder is not None:
            return self._embedder
        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder
            backend = (self.config.embed_backend or "").strip().lower()
            if backend == "hash":
                self._embedder = _HashingEmbedder()
                return self._embedder
            try:
                self._embedder = _SentenceTransformerEmbedder(self.config.embed_model)
                return self._embedder
            except Exception:
                pass
            self._embedder = _HashingEmbedder()
            return self._embedder

    def warmup_embedder(self) -> None:
        """Load the embedding model at startup so the first request does not time out."""
        _ = self._get_embedder()
