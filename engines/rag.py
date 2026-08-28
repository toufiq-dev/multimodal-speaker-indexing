"""Speaker-Aware RAG for Bengali talk-show transcripts.

Chapter 6.3.1 extension: solves unresolved-name UX by retrieving
speaker-attributed segments with citations. Uses multilingual MiniLM
for Bengali/English code-switch robustness and ChromaDB for local
vector store (Kaggle-safe, no heavy deps beyond sentence-transformers).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import config
from models import FinalSegment
from runtime import release_gpu_memory

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    _CHROMA_AVAILABLE = True
except Exception:
    chromadb = None  # type: ignore
    _CHROMA_AVAILABLE = False


CHUNK_MIN = 400
CHUNK_MAX = 600
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _chunk_segments(segments: List[FinalSegment], min_chars: int = CHUNK_MIN, max_chars: int = CHUNK_MAX) -> List[Dict]:
    """Chunk FinalSegments into 400-600 char windows preserving speaker metadata."""
    chunks: List[Dict] = []
    for seg in segments:
        text = seg.text or ""
        if not text:
            continue
        # If short, keep as one chunk
        if len(text) <= max_chars:
            chunks.append({
                "text": text,
                "speaker": seg.speaker,
                "start_time": seg.start,
                "end_time": seg.end,
                "fusion_confidence": seg.confidence,
            })
            continue
        # Long: split on danda/boundary near max
        words = text.split()
        cur_words: List[str] = []
        cur_len = 0
        for w in words:
            cur_words.append(w)
            cur_len += len(w) + 1
            if cur_len >= min_chars:
                # try to cut at sentence end
                chunk_text = " ".join(cur_words)
                # If next words would exceed max, flush now
                if cur_len >= max_chars or w.endswith("।") or w.endswith("."):
                    chunks.append({
                        "text": chunk_text,
                        "speaker": seg.speaker,
                        "start_time": seg.start,
                        "end_time": seg.end,
                        "fusion_confidence": seg.confidence,
                    })
                    cur_words = []
                    cur_len = 0
        if cur_words:
            chunks.append({
                "text": " ".join(cur_words),
                "speaker": seg.speaker,
                "start_time": seg.start,
                "end_time": seg.end,
                "fusion_confidence": seg.confidence,
            })
    return chunks


class SpeakerAwareRAG:
    """ Speaker-aware retrieval over FinalSegments.

    Ingests result.json (list of FinalSegment), chunks, embeds, stores in
    ChromaDB in-memory (or persistent if persist_dir given). Query returns
    top-k with speaker filter and citation.
    """

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL, persist_dir: Optional[str] = None):
        self.embed_model_name = embed_model
        self.persist_dir = persist_dir
        self._embedder = None
        self._client = None
        self._collection = None
        if _ST_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(
                    embed_model, device="cuda" if config.use_cuda() else "cpu")
            except Exception as e:
                print(f"[rag] embedder unavailable ({e.__class__.__name__}: {e}); "
                      f"retrieval degrades to keyword overlap")
                self._embedder = None
        if _CHROMA_AVAILABLE:
            try:
                if persist_dir:
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
                else:
                    self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
            except Exception:
                self._client = None
        # Fallback in-memory lists if chroma/embedder missing
        self._fallback_chunks: List[Dict] = []
        self._fallback_embs = None

    def _embed(self, texts: List[str]):
        if self._embedder is None:
            # Fallback: dummy zero embeddings (keyword search fallback)
            import numpy as np
            return np.zeros((len(texts), 384), dtype=np.float32)
        return self._embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False)

    def ingest(self, segments: List[FinalSegment] | str | Path, collection_name: str = "talkshow") -> int:
        """Ingest from list of FinalSegment or path to result.json. Returns chunk count."""
        if isinstance(segments, (str, Path)):
            p = Path(segments)
            data = json.loads(p.read_text(encoding="utf-8"))
            segs = [FinalSegment(start=d["start"], end=d["end"], speaker=d["speaker"], text=d["text"], confidence=d.get("confidence", 0.0)) for d in data]
        else:
            segs = segments
        chunks = _chunk_segments(segs)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        # Chroma path
        if self._client is not None and self._embedder is not None:
            try:
                # recreate collection
                try:
                    self._client.delete_collection(collection_name)
                except Exception:
                    pass
                self._collection = self._client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
                embs = self._embed(texts)
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"speaker": c["speaker"], "start_time": float(c["start_time"]), "end_time": float(c["end_time"]), "fusion_confidence": float(c["fusion_confidence"])} for c in chunks]
                self._collection.add(ids=ids, documents=texts, embeddings=embs.tolist() if hasattr(embs, "tolist") else embs, metadatas=metadatas)
                release_gpu_memory()
                return len(chunks)
            except Exception:
                pass
        # Fallback in-memory
        self._fallback_chunks = chunks
        try:
            self._fallback_embs = self._embed(texts)
        except Exception:
            self._fallback_embs = None
        release_gpu_memory()
        return len(chunks)

    def query(self, question: str, speaker_filter: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Return top-k chunks with citations. speaker_filter optional (e.g., 'Abu Hena Razzaki')."""
        if not question:
            return []
        # Chroma path
        if self._collection is not None and self._embedder is not None:
            try:
                q_emb = self._embed([question])
                where = {"speaker": speaker_filter} if speaker_filter else None
                res = self._collection.query(query_embeddings=q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb, n_results=k, where=where)
                out = []
                if res and res.get("documents"):
                    docs = res["documents"][0]
                    metas = res["metadatas"][0]
                    dists = res.get("distances", [[0]*len(docs)])[0]
                    for doc, meta, dist in zip(docs, metas, dists):
                        out.append({"text": doc, "speaker": meta.get("speaker"), "start_time": meta.get("start_time"), "end_time": meta.get("end_time"), "fusion_confidence": meta.get("fusion_confidence"), "score": 1 - float(dist) if dist is not None else 0.0, "citation": f"[{meta.get('start_time'):.1f}s–{meta.get('end_time'):.1f}s {meta.get('speaker')}]"})
                return out
            except Exception:
                pass
        # Fallback: brute cosine over fallback embeddings or keyword overlap
        if self._fallback_chunks and self._fallback_embs is not None:
            try:
                import numpy as np
                q_emb = self._embed([question])[0]
                # cosine = dot (normalized)
                sims = (self._fallback_embs @ q_emb) if self._fallback_embs.ndim == 2 else np.array([0]*len(self._fallback_chunks))
                # speaker filter
                idxs = list(range(len(self._fallback_chunks)))
                if speaker_filter:
                    idxs = [i for i in idxs if self._fallback_chunks[i]["speaker"] == speaker_filter]
                    if not idxs:
                        idxs = list(range(len(self._fallback_chunks)))
                # top-k
                scored = sorted([(i, float(sims[i]) if i < len(sims) else 0.0) for i in idxs], key=lambda x: -x[1])
                out = []
                for i, s in scored[:k]:
                    c = self._fallback_chunks[i]
                    out.append({"text": c["text"], "speaker": c["speaker"], "start_time": c["start_time"], "end_time": c["end_time"], "fusion_confidence": c["fusion_confidence"], "score": s, "citation": f"[{c['start_time']:.1f}s–{c['end_time']:.1f}s {c['speaker']}]"})
                return out
            except Exception:
                pass
        # Keyword fallback
        q_words = set(question.lower().split())
        scored = []
        for c in self._fallback_chunks:
            overlap = len(q_words & set(c["text"].lower().split()))
            if speaker_filter and c["speaker"] != speaker_filter:
                overlap *= 0.5
            scored.append((c, overlap))
        scored.sort(key=lambda x: -x[1])
        return [{"text": c["text"], "speaker": c["speaker"], "start_time": c["start_time"], "end_time": c["end_time"], "fusion_confidence": c["fusion_confidence"], "score": float(s), "citation": f"[{c['start_time']:.1f}s–{c['end_time']:.1f}s {c['speaker']}]"} for c, s in scored[:k]]

    def close(self) -> None:
        """Drop the embedder and return its device memory."""
        self._embedder = None
        self._collection = None
        self._client = None
        release_gpu_memory()
