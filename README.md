# PubMed LLM Pipeline

An end-to-end biomedical literature mining pipeline for retrieving PubMed records, extracting structured scientific entities with an LLM, validating the output with typed schemas, and producing a machine-readable research synthesis.

This repository is intentionally compact: the core production path lives in `pipeline.py`, while `app.py` exposes the same workflow through a Gradio interface. The project is designed as a practical foundation for biomedical text mining, systematic review acceleration, target discovery workflows, competitive intelligence, and downstream bioinformatics automation.

## System Architecture and Core Concept

The PubMed LLM Pipeline automates a common scientific workflow:

1. Search PubMed for literature matching a biomedical query.
2. Fetch article metadata and abstracts from NCBI.
3. Normalize each paper into a consistent internal representation.
4. Ask an LLM to extract structured entities from each abstract.
5. Validate the LLM response against strict Pydantic schemas.
6. Cache per-paper extraction results to avoid repeated model calls.
7. Synthesize cross-paper insights into a concise report.
8. Write a reproducible JSON artifact for downstream analysis.

In plain English, this pipeline turns unstructured biomedical abstracts into structured scientific data. Instead of asking a researcher to manually read papers and record proteins, methods, datasets, and strategic takeaways, the pipeline performs a deterministic PubMed retrieval step followed by schema-constrained LLM reasoning.

### Why LLM-assisted extraction?

Traditional biomedical extraction workflows often rely on regular expressions, keyword dictionaries, or fully manual curation. Those approaches are useful when terminology is stable and the desired facts are simple, but biomedical literature is not written in a single predictable format. A method may be described by acronym, full name, variant spelling, or contextual phrasing. A dataset may appear only as part of an experimental description. A protein mention may be embedded in a dense sentence with assay details, disease context, and model assumptions.

LLMs add value because they can reason over local context, map semantically equivalent phrasing to the same concept class, and extract higher-level relationships that are difficult to encode with static rules. This repository still keeps the reliable parts deterministic: PubMed retrieval is handled by NCBI E-utilities, data shape is enforced by Pydantic, and output is serialized as JSON. The LLM is used where language understanding matters most.

## Technical Specifications and Under-the-Hood Mechanics

### Runtime overview

The main entry point is:

```python
from pipeline import run_pipeline

result = run_pipeline(
    query="protein folding neural network",
    n=50,
    output="report.json",
)
```

At runtime, `run_pipeline()` performs the following sequence:

1. Creates the local extraction cache directory.
2. Normalizes the PubMed query when needed.
3. Retrieves PubMed IDs with Entrez `esearch`.
4. Fetches PubMed XML records with Entrez `efetch`.
5. Parses metadata, titles, abstracts, PMIDs, and DOIs.
6. Loads cached entity extractions when available.
7. Calls Claude through the Anthropic API for uncached papers.
8. Validates extracted entities with `PaperEntities`.
9. Produces a five-bullet synthesis validated by `CorpusSynthesis`.
10. Writes a JSON report when an output path is supplied.

### Data ingestion: NCBI Entrez and PubMed E-utilities

The ingestion layer uses Biopython's `Bio.Entrez` client, which wraps the NCBI E-utilities API.

Implemented E-utilities:

- `esearch`: Searches the PubMed database and returns matching PubMed IDs.
- `efetch`: Retrieves PubMed records for those IDs in XML mode.

Key implementation details:

- `fetch_pubmed_ids(query, n)` normalizes user input, calls `Entrez.esearch(db="pubmed", ...)`, and paginates through results using `retstart` and `retmax`.
- `fetch_pubmed_records(ids)` batches PubMed IDs into chunks of up to 100 records per `efetch` call.
- `extract_record(article)` converts Entrez XML objects into a compact dictionary containing `pmid`, `title`, `abstract`, and `doi`.
- `retry_call()` retries transient failures such as rate limits, gateway failures, timeout errors, and connection errors with exponential backoff.

The code currently limits PubMed ID search pages to 200 records and fetches metadata in batches of 100 IDs. This is a practical request-size control strategy for PubMed XML retrieval. For larger production workloads, this layer can be extended with explicit global rate limiting, queue-based ingestion, persistent raw XML storage, and NCBI-compliant request scheduling.

### Query normalization

`normalize_pubmed_query()` contains a targeted normalization rule for queries containing the standalone term `AI`. It expands that term into a PubMed title/abstract query:

```text
("artificial intelligence"[Title/Abstract] OR AI[Title/Abstract])
```

When the remaining query is plain text, the function scopes it to `Title/Abstract` as well. This keeps common natural-language queries closer to explicit PubMed search syntax while preserving advanced user-provided boolean queries.

### LLM and orchestration layer

The repository uses the Anthropic Python SDK for LLM inference. The default model is configured through:

```text
ANTHROPIC_MODEL=claude-haiku-4-5
```

The current implementation does not use embeddings, FAISS, SentenceTransformers, or a separate vector database. It sends each paper's title and abstract directly into the model context window for entity extraction, then sends the complete extracted paper collection into a synthesis prompt.

This is a direct-context architecture rather than an embedding-based RAG architecture:

- Retrieval is deterministic PubMed retrieval through Entrez.
- Context construction is performed directly from fetched titles and abstracts.
- LLM output is constrained by Pydantic schemas.
- Caching prevents repeated extraction for the same PMID or DOI.

The pipeline can be extended into a semantic-vector RAG system by adding a chunking layer, embedding model, vector index, and citation-aware retrieval step before synthesis. The current code keeps the first production version simpler and more auditable.

### Prompting and structured output

Entity extraction is handled by `extract_entities()`:

```python
prompt = (
    f"Title: {paper.get('title', '')}\n"
    f"Abstract: {paper.get('abstract', '')}\n\n"
    "Extract named proteins, experimental/computational methods, and datasets."
)
```

The system prompt frames the model as a precise biomedical information extraction system. The output is validated against:

```python
class PaperEntities(BaseModel):
    proteins: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
```

Corpus-level synthesis is handled by `synthesize()` and validated against:

```python
class CorpusSynthesis(BaseModel):
    synthesis_bullets: list[str] = Field(min_length=5, max_length=5)
```

The LLM call path first attempts Anthropic structured parsing through `client.messages.parse(...)` with `output_format=schema`. If the installed SDK does not support that interface, it falls back to `client.messages.create(...)` and instructs the model to return JSON matching the schema. `parsed_payload()` then extracts and validates the JSON object.

### Schema validation and hallucination control

The pipeline does not assume that model output is trustworthy merely because it is fluent. Every LLM response is passed through Pydantic validation before entering the report.

Validation provides several guardrails:

- Entity extraction must produce the exact fields expected by `PaperEntities`.
- Corpus synthesis must produce exactly five bullets through `min_length=5` and `max_length=5`.
- Invalid JSON, missing JSON, malformed fields, or incompatible types fail before they can be serialized as final output.
- Cached files are revalidated when loaded, preventing corrupted cache artifacts from silently contaminating downstream runs.

This approach does not eliminate all semantic hallucination risk, but it prevents structural hallucination and makes the output suitable for downstream bioinformatics workflows that expect stable machine-readable contracts.

### Caching strategy

LLM extraction results are cached in `.extracted_cache/`.

Cache keys are based on:

- PMID when available.
- SHA-256 hash of DOI when a DOI is available.

`load_cached()` validates cached JSON with `PaperEntities.model_validate_json()` before reuse. `save_cached()` writes only validated entity payloads. This design reduces API cost, improves repeatability, and allows repeated synthesis experiments without re-extracting every article.

### Output contract

`run_pipeline()` returns and optionally writes a JSON document with this shape:

```json
{
  "query": "ai drug discovery",
  "count": 1,
  "synthesis": [
    "Five validated synthesis bullets..."
  ],
  "papers": {
    "42381135": {
      "metadata": {
        "pmid": "42381135",
        "title": "Article title",
        "abstract": "Article abstract",
        "doi": "10.xxxx/example"
      },
      "entities": {
        "proteins": [],
        "methods": [],
        "datasets": []
      }
    }
  }
}
```

This structure is intentionally friendly to notebooks, ETL jobs, dashboard backends, knowledge graph loaders, and downstream statistical analysis.

## Repository Architecture and File Structure

```text
pubmed-llm-pipeline/
|-- README.md
|-- pipeline.py
|-- app.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- report.json
|-- .extracted_cache/
`-- .venv/
```

Path descriptions:

- `pipeline.py`: Core ingestion, extraction, validation, caching, synthesis, and CLI orchestration logic.
- `app.py`: Gradio application that exposes the pipeline through a browser-based JSON interface.
- `requirements.txt`: Python package dependencies, including Anthropic, Biopython, Gradio, and Pydantic.
- `.env.example`: Template for required runtime environment variables.
- `.env`: Local environment file for API keys and model configuration. This file is ignored by Git.
- `.extracted_cache/`: Local cache for validated per-paper LLM extraction results. This directory is ignored by Git.
- `report.json`: Example generated output artifact. Future generated reports should generally remain uncommitted.
- `.gitignore`: Prevents local secrets, virtual environments, caches, bytecode, and generated reports from being committed.

Although the current repository is intentionally minimal, larger deployments commonly split this structure into dedicated modules such as `ingestion/`, `schemas/`, `prompts/`, `llm/`, `cache/`, and `outputs/`. The current file layout favors readability and fast iteration.

