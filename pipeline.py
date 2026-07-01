"""PubMed text-mining pipeline with local caching and Anthropic extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from Bio import Entrez
from Bio.Entrez.Parser import ValidationError
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError


Entrez.email = "your.email@example.com"
Entrez.api_key = os.environ.get("NCBI_API_KEY") or "16f45c16cb287905a3bb04dbaf57460ccd09"

CACHE_DIR = Path(".extracted_cache")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
T = TypeVar("T", bound=BaseModel)


class PaperEntities(BaseModel):
    proteins: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)


class CorpusSynthesis(BaseModel):
    synthesis_bullets: list[str] = Field(min_length=5, max_length=5)


def retry_call(fn, *, attempts: int = 4, base_delay: float = 1.5) -> Any:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in {429, 500, 502, 503, 504} or isinstance(
                exc, (TimeoutError, ConnectionError)
            )
            if attempt == attempts - 1 or not retryable:
                raise
            time.sleep(base_delay * (2**attempt))


def fetch_pubmed_ids(query: str, n: int) -> list[str]:
    ids: list[str] = []
    page_size = min(200, max(1, n))
    for start in range(0, n, page_size):
        handle = retry_call(
            lambda: Entrez.esearch(
                db="pubmed", term=query, retmax=min(page_size, n - start), retstart=start
            )
        )
        try:
            record = Entrez.read(handle)
        finally:
            handle.close()
        ids.extend(record.get("IdList", []))
        if len(record.get("IdList", [])) < page_size:
            break
    return ids[:n]


def extract_record(article: dict[str, Any]) -> dict[str, str]:
    citation = article.get("MedlineCitation", {})
    pmid = str(citation.get("PMID", ""))
    article_data = citation.get("Article", {})
    title = str(article_data.get("ArticleTitle", "") or "")
    abstract_parts = article_data.get("Abstract", {}).get("AbstractText", []) or []
    abstract = " ".join(str(part) for part in abstract_parts if part)
    doi = ""
    for item in article.get("PubmedData", {}).get("ArticleIdList", []) or []:
        if getattr(item, "attributes", {}).get("IdType") == "doi":
            doi = str(item)
            break
    return {"pmid": pmid, "title": title, "abstract": abstract, "doi": doi}


def fetch_pubmed_records(ids: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for start in range(0, len(ids), 100):
        chunk = ids[start : start + 100]
        if not chunk:
            continue
        handle = retry_call(
            lambda: Entrez.efetch(db="pubmed", id=",".join(chunk), retmode="xml")
        )
        try:
            data = Entrez.read(handle)
        except (ValidationError, ValueError, KeyError, TypeError):
            data = []
        finally:
            handle.close()
        records.extend(extract_record(article) for article in data.get("PubmedArticle", []))
    return records


def doi_hash(doi: str) -> str:
    return hashlib.sha256(doi.strip().lower().encode("utf-8")).hexdigest()


def cache_paths(paper: dict[str, str]) -> list[Path]:
    paths = []
    if paper.get("pmid"):
        paths.append(CACHE_DIR / f"{paper['pmid']}.json")
    if paper.get("doi"):
        paths.append(CACHE_DIR / f"{doi_hash(paper['doi'])}.json")
    return paths


def load_cached(paper: dict[str, str]) -> PaperEntities | None:
    for path in cache_paths(paper):
        if path.exists():
            try:
                return PaperEntities.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, PydanticValidationError, json.JSONDecodeError):
                continue
    return None


def save_cached(paper: dict[str, str], entities: PaperEntities) -> None:
    paths = cache_paths(paper)
    if not paths:
        return
    CACHE_DIR.mkdir(exist_ok=True)
    paths[0].write_text(entities.model_dump_json(indent=2), encoding="utf-8")


def make_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for uncached Anthropic calls.")
    return anthropic.Anthropic(api_key=api_key)


def parsed_payload(response: Any, schema: type[T]) -> T:
    candidate = getattr(response, "parsed", None)
    if candidate is None and getattr(response, "content", None):
        candidate = getattr(response.content[0], "parsed", None)
    if candidate is not None:
        return schema.model_validate(candidate)
    text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Anthropic response did not contain a JSON object.")
    return schema.model_validate_json(text[start : end + 1])


def call_claude(client: anthropic.Anthropic, schema: type[T], system: str, prompt: str) -> T:
    def request() -> T:
        try:
            response = client.messages.parse(
                model=MODEL,
                max_tokens=1200,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
        except AttributeError:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1200,
                system=f"{system}\nReturn only valid JSON matching this schema: {schema.model_json_schema()}",
                messages=[{"role": "user", "content": prompt}],
            )
        return parsed_payload(response, schema)

    return retry_call(request)


def extract_entities(client: anthropic.Anthropic, paper: dict[str, str]) -> PaperEntities:
    cached = load_cached(paper)
    if cached:
        return cached
    prompt = (
        f"Title: {paper.get('title', '')}\n"
        f"Abstract: {paper.get('abstract', '')}\n\n"
        "Extract named proteins, experimental/computational methods, and datasets."
    )
    entities = call_claude(
        client,
        PaperEntities,
        "You are a precise biomedical information extraction system.",
        prompt,
    )
    save_cached(paper, entities)
    return entities


def synthesize(client: anthropic.Anthropic, papers: dict[str, Any]) -> CorpusSynthesis:
    prompt = (
        "Create exactly five dense, high-impact cross-paper takeaways from this "
        f"text-mined collection:\n{json.dumps(papers, ensure_ascii=True)}"
    )
    return call_claude(
        client,
        CorpusSynthesis,
        "You synthesize biomedical literature into concise strategic findings.",
        prompt,
    )


def run_pipeline(query: str, n: int = 50, output: str | None = "report.json") -> dict[str, Any]:
    CACHE_DIR.mkdir(exist_ok=True)
    client: anthropic.Anthropic | None = None
    ids = fetch_pubmed_ids(query, n)
    records = fetch_pubmed_records(ids)
    per_paper: dict[str, Any] = {}
    for paper in records:
        key = paper.get("pmid") or paper.get("doi") or hashlib.sha1(
            json.dumps(paper, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = load_cached(paper)
        if cached:
            entities = cached
        else:
            client = client or make_anthropic_client()
            entities = extract_entities(client, paper)
        per_paper[key] = {"metadata": paper, "entities": entities.model_dump()}
    synthesis = (
        synthesize(client or make_anthropic_client(), per_paper).synthesis_bullets
        if per_paper
        else []
    )
    result = {"query": query, "count": len(per_paper), "synthesis": synthesis, "papers": per_paper}
    if output:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine PubMed papers with Claude.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.query, args.n, args.output), indent=2))


if __name__ == "__main__":
    main()
