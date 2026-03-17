from __future__ import annotations

"""
TBS category-level RAG attribute extraction pipeline.

Quick start:
  1) Configure API keys in .env (optional) and edit configs/tbs_rag.yaml
  2) Run:
       python scripts/tbs_rag_pipeline.py --config configs/tbs_rag.yaml

Steps:
  parse   -> extract per-page text from PDF
  chunk   -> clean + chunk text
  embed   -> build embeddings + FAISS index
  retrieve-> run category queries
  generate-> LLM attribute generation + validation
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import fitz  # pymupdf
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: pymupdf (fitz). Install pymupdf.") from exc

try:
    import faiss
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: faiss-cpu. Install faiss-cpu.") from exc


DEFAULT_CATEGORIES = ["ASC-US", "ASC-H", "LSIL", "HSIL", "NILM"]
QUERY_TEMPLATE = (
    "Retrieve the key morphological features of category {k} according to the TBS system and pathology guidelines."
)
LLM_PROMPT_TEMPLATE = """Based only on the retrieved excerpts from The Bethesda System for Reporting Cervical Cytology (TBS), list the key morphological attributes of category {k}.

Requirements:
1. Use only information supported by the retrieved text.
2. Focus on clinically relevant cytomorphological features.
3. Return 5 to 12 short attribute phrases.
4. Do not include explanations.
5. Output JSON only in the following format:

{{
  "category": "{k}",
  "attributes": ["attribute 1", "attribute 2", "..."]
}}

Retrieved excerpts:
{context}
"""


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Config root must be a mapping")
    return data


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def setup_logger(log_path: Optional[Path]):
    def _log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if log_path:
            ensure_parent(log_path)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    return _log


def resolve_api_key(cfg: Dict[str, Any]) -> Optional[str]:
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if api_key_env:
        return os.getenv(str(api_key_env))
    if isinstance(api_key, str) and api_key.lower().startswith("env:"):
        return os.getenv(api_key.split(":", 1)[1].strip())
    return api_key


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"(?i)page\s*\d+", raw) or re.fullmatch(r"\d{1,4}", raw):
            continue
        cleaned_lines.append(raw)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"-\n(?=[a-z])", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def parse_pdf_pages(pdf_path: Path, logger) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if not text.strip():
            logger(f"warning: page {i + 1} empty")
        pages.append({"page_num": i + 1, "text": text})
    return pages


def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_pages = []
    for page in pages:
        cleaned_pages.append(
            {"page_num": page["page_num"], "text": clean_text(page.get("text", ""))}
        )
    return cleaned_pages


def build_corpus(pages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, int]]]:
    parts: List[str] = []
    offsets: List[Dict[str, int]] = []
    cursor = 0
    for page in pages:
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        text = page.get("text", "")
        parts.append(text)
        cursor += len(text)
        end = cursor
        offsets.append({"page_num": int(page["page_num"]), "start": start, "end": end})
    return "".join(parts), offsets


def find_page_span(
    offsets: List[Dict[str, int]], start: int, end: int
) -> Tuple[int, int]:
    page_start = offsets[0]["page_num"]
    page_end = offsets[-1]["page_num"]
    for item in offsets:
        if end <= item["start"]:
            break
        if start < item["end"]:
            page_start = item["page_num"]
            break
    for item in reversed(offsets):
        if start >= item["end"]:
            break
        if end > item["start"]:
            page_end = item["page_num"]
            break
    return page_start, page_end


def chunk_corpus(
    corpus: str,
    offsets: List[Dict[str, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: List[Dict[str, Any]] = []
    step = chunk_size - chunk_overlap
    chunk_id = 0
    for start in range(0, len(corpus), step):
        end = min(start + chunk_size, len(corpus))
        text = corpus[start:end].strip()
        if not text:
            continue
        page_start, page_end = find_page_span(offsets, start, end)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
                "source": "TBS",
            }
        )
        chunk_id += 1
    return chunks


class ThirdPartyEmbeddingAPI:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Embedding API failed after retries: {last_err}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "input": texts}
        data = self._post(payload)
        if "data" not in data:
            raise RuntimeError("Embedding API response missing data field")
        items = data["data"]
        if items and isinstance(items[0], dict) and "embedding" in items[0]:
            items = sorted(items, key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        if items and isinstance(items[0], list):
            return items
        raise RuntimeError("Unexpected embedding response format")

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


@dataclass
class LLMResponse:
    raw: str
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"LLM API failed after retries: {last_err}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a pathology assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        data = self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM response missing content")
        return content


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def validate_attributes(category: str, payload: Dict[str, Any]) -> List[str]:
    if payload.get("category") != category:
        raise ValueError("category mismatch")
    attrs = payload.get("attributes")
    if not isinstance(attrs, list):
        raise ValueError("attributes must be list")
    cleaned: List[str] = []
    seen = set()
    for item in attrs:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    if not 3 <= len(cleaned) <= 15:
        raise ValueError("attributes length out of range")
    return cleaned


def build_retrieval_queries(categories: List[str]) -> Dict[str, str]:
    return {cat: QUERY_TEMPLATE.format(k=cat) for cat in categories}


def save_config_snapshot(cfg: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    snapshot = dict(cfg)
    snapshot["run_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[p{chunk['page_start']}-{chunk['page_end']}]"
        parts.append(f"{header} {chunk['text']}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="TBS RAG attribute extraction pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated steps: parse,chunk,embed,retrieve,generate or all",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser.add_argument("--limit-pages", type=int, default=0, help="Limit pages for quick tests")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Limit chunks for quick tests")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(Path(args.config))
    data_cfg = cfg.get("data", {})
    chunk_cfg = cfg.get("chunking", {})
    retrieval_cfg = cfg.get("retrieval", {})
    embed_cfg = cfg.get("embedding_api", {})
    llm_cfg = cfg.get("llm", {})
    categories = cfg.get("categories") or DEFAULT_CATEGORIES
    if not isinstance(categories, list):
        raise SystemExit("categories must be a list")

    log_path_raw = data_cfg.get("run_log_path")
    log_path = Path(log_path_raw) if log_path_raw else None
    logger = setup_logger(log_path)

    steps_arg = [s.strip() for s in args.steps.split(",") if s.strip()]
    steps = (
        ["parse", "chunk", "embed", "retrieve", "generate"]
        if "all" in steps_arg
        else steps_arg
    )

    pdf_path = Path(data_cfg.get("pdf_path"))
    parsed_pages_path = Path(data_cfg.get("parsed_pages_path"))
    chunks_path = Path(data_cfg.get("chunks_path"))
    chunk_metadata_path = Path(data_cfg.get("chunk_metadata_path"))
    faiss_index_path = Path(data_cfg.get("faiss_index_path"))
    retrieval_results_path = Path(data_cfg.get("retrieval_results_path"))
    attribute_sets_raw_path = Path(data_cfg.get("attribute_sets_raw_path"))
    attribute_sets_path = Path(data_cfg.get("attribute_sets_path"))
    run_config_raw = data_cfg.get("run_config_path")
    run_config_path = Path(run_config_raw) if run_config_raw else None

    if run_config_path:
        save_config_snapshot(cfg, run_config_path)

    pages: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    if "parse" in steps:
        if parsed_pages_path.exists() and not args.force:
            logger(f"skip parse: {parsed_pages_path} exists")
        else:
            logger(f"parsing PDF: {pdf_path}")
            pages = parse_pdf_pages(pdf_path, logger)
            if args.limit_pages:
                pages = pages[: args.limit_pages]
            write_jsonl(parsed_pages_path, pages)
            logger(f"saved parsed pages -> {parsed_pages_path}")

    if "chunk" in steps:
        if not pages:
            pages = read_jsonl(parsed_pages_path)
            if args.limit_pages:
                pages = pages[: args.limit_pages]
        if chunks_path.exists() and chunk_metadata_path.exists() and not args.force:
            logger(f"skip chunk: {chunks_path} exists")
        else:
            logger("cleaning + chunking text")
            cleaned_pages = clean_pages(pages)
            corpus, offsets = build_corpus(cleaned_pages)
            chunk_size = int(chunk_cfg.get("chunk_size", 1000))
            chunk_overlap = int(chunk_cfg.get("chunk_overlap", 150))
            chunks = chunk_corpus(corpus, offsets, chunk_size, chunk_overlap)
            if args.limit_chunks:
                chunks = chunks[: args.limit_chunks]
            write_jsonl(chunks_path, chunks)
            ensure_parent(chunk_metadata_path)
            chunk_metadata_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger(f"saved chunks -> {chunks_path}")

    if "embed" in steps:
        if not chunks:
            if chunks_path.exists():
                chunks = read_jsonl(chunks_path)
            else:
                chunks = json.loads(chunk_metadata_path.read_text(encoding="utf-8"))
        if faiss_index_path.exists() and not args.force:
            logger(f"skip embed: {faiss_index_path} exists")
        else:
            api_key = resolve_api_key(embed_cfg)
            if not api_key:
                raise SystemExit("embedding_api.api_key missing")
            embedder = ThirdPartyEmbeddingAPI(
                base_url=str(embed_cfg.get("base_url")),
                api_key=api_key,
                model_name=str(embed_cfg.get("model_name")),
                timeout=int(embed_cfg.get("timeout", 60)),
                max_retries=int(embed_cfg.get("max_retries", 3)),
            )
            texts = [c["text"] for c in chunks]
            batch_size = int(embed_cfg.get("batch_size", 32))
            embeddings: List[List[float]] = []
            logger(f"embedding {len(texts)} chunks (batch_size={batch_size})")
            for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
                batch = texts[i : i + batch_size]
                embeddings.extend(embedder.embed_texts(batch))
            vectors = np.array(embeddings, dtype="float32")
            normalize = bool(retrieval_cfg.get("normalize_embeddings", True))
            metric = str(retrieval_cfg.get("metric", "cosine")).lower()
            if normalize or metric == "cosine":
                faiss.normalize_L2(vectors)
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            ensure_parent(faiss_index_path)
            faiss.write_index(index, str(faiss_index_path))
            logger(f"saved faiss index -> {faiss_index_path}")

    if "retrieve" in steps:
        if not chunks:
            if chunks_path.exists():
                chunks = read_jsonl(chunks_path)
            else:
                chunks = json.loads(chunk_metadata_path.read_text(encoding="utf-8"))
        if retrieval_results_path.exists() and not args.force:
            logger(f"skip retrieve: {retrieval_results_path} exists")
        else:
            api_key = resolve_api_key(embed_cfg)
            if not api_key:
                raise SystemExit("embedding_api.api_key missing")
            embedder = ThirdPartyEmbeddingAPI(
                base_url=str(embed_cfg.get("base_url")),
                api_key=api_key,
                model_name=str(embed_cfg.get("model_name")),
                timeout=int(embed_cfg.get("timeout", 60)),
                max_retries=int(embed_cfg.get("max_retries", 3)),
            )
            index = faiss.read_index(str(faiss_index_path))
            top_k = int(retrieval_cfg.get("top_k", 5))
            queries = build_retrieval_queries(categories)
            normalize = bool(retrieval_cfg.get("normalize_embeddings", True))
            results: Dict[str, Any] = {}
            for cat, query in queries.items():
                qvec = np.array([embedder.embed_query(query)], dtype="float32")
                if normalize:
                    faiss.normalize_L2(qvec)
                scores, indices = index.search(qvec, top_k)
                hits = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    chunk = chunks[int(idx)]
                    hits.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "score": float(score),
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "text": chunk["text"],
                        }
                    )
                results[cat] = {"query": query, "top_k": top_k, "hits": hits}
            ensure_parent(retrieval_results_path)
            retrieval_results_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger(f"saved retrieval results -> {retrieval_results_path}")

    if "generate" in steps:
        if not retrieval_results_path.exists():
            raise SystemExit("retrieval_results.json missing; run retrieve step first")
        if attribute_sets_path.exists() and not args.force:
            logger(f"skip generate: {attribute_sets_path} exists")
        else:
            api_key = resolve_api_key(llm_cfg)
            if not api_key:
                raise SystemExit("llm.api_key missing")
            llm = OpenAICompatibleLLM(
                base_url=str(llm_cfg.get("base_url")),
                api_key=api_key,
                model_name=str(llm_cfg.get("model_name")),
                temperature=float(llm_cfg.get("temperature", 0)),
                timeout=int(llm_cfg.get("timeout", 60)),
                max_retries=int(llm_cfg.get("max_retries", 2)),
            )
            retrieval = json.loads(retrieval_results_path.read_text(encoding="utf-8"))
            raw_outputs: Dict[str, Any] = {}
            cleaned_outputs: Dict[str, List[str]] = {}
            for cat in categories:
                hits = retrieval.get(cat, {}).get("hits", [])
                context = build_context(hits)
                prompt = LLM_PROMPT_TEMPLATE.format(k=cat, context=context)
                logger(f"LLM generating attributes for {cat}")
                raw_text = llm.generate(prompt)
                parsed: Optional[Dict[str, Any]] = None
                error: Optional[str] = None
                try:
                    parsed = parse_llm_json(raw_text)
                    attrs = validate_attributes(cat, parsed)
                    cleaned_outputs[cat] = attrs
                except Exception as exc:
                    error = str(exc)
                    retry_prompt = prompt + "\n\nReturn valid JSON only."
                    try:
                        raw_text_retry = llm.generate(retry_prompt)
                        parsed = parse_llm_json(raw_text_retry)
                        attrs = validate_attributes(cat, parsed)
                        cleaned_outputs[cat] = attrs
                        raw_text = raw_text_retry
                        error = None
                    except Exception as exc_retry:
                        error = f"{error} | retry_failed: {exc_retry}"
                raw_outputs[cat] = {"raw": raw_text, "parsed": parsed, "error": error}
            ensure_parent(attribute_sets_raw_path)
            attribute_sets_raw_path.write_text(
                json.dumps(raw_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ensure_parent(attribute_sets_path)
            attribute_sets_path.write_text(
                json.dumps(cleaned_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger(f"saved attribute sets -> {attribute_sets_path}")


if __name__ == "__main__":
    main()
