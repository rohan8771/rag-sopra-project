"""
legacy_backend_ingestion.py

Purpose:
This script builds TWO searchable knowledge layers for our dummy legacy backend.

Layer 1:
    Raw original chunks
    Example: actual Java code, docs text, application.properties content.

Layer 2:
    Structured modernization records produced by LangExtract
    Example: "Billing module uses JDBC Statement with SQL string concatenation."

Why both?
    Raw chunks are good for evidence and exact source code.
    Structured records are good for modernization summaries, risk lists,
    technology inventories, dependency maps, etc.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

import langextract as lx

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# 1. Basic paths
# ---------------------------------------------------------------------

# This script is expected to live in the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

# This is the fake legacy codebase we created.
LEGACY_BACKEND_ROOT = PROJECT_ROOT / "legacy_backend"

# We keep this separate from your existing LangChain docs vector DB.
# This avoids polluting your already-working chroma_db.
LEGACY_CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# We also save structured extraction records as JSON for inspection.
OUTPUT_DIR = PROJECT_ROOT / "legacy_backend_extracted_records"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "modernization_records.jsonl"


# ---------------------------------------------------------------------
# 2. Model / embedding configuration
# ---------------------------------------------------------------------

# Load .env so OPENAI_API_KEY 
load_dotenv()

# Same embedding model style as your earlier RAG app.
EMBEDDING_MODEL = "text-embedding-3-small"

# LangExtract model.
# You can later change this to gpt-5-mini if you want.
LANGEXTRACT_MODEL_ID = "gpt-4o-mini"


# ---------------------------------------------------------------------
# 3. File types we want to ingest
# ---------------------------------------------------------------------

# Keep this small and explicit for now.
# We only ingest files useful for modernization analysis.
SUPPORTED_EXTENSIONS = {
    ".java",
    ".txt",
    ".properties",
}


# ---------------------------------------------------------------------
# 4. LangExtract prompt and examples
# ---------------------------------------------------------------------

MODERNIZATION_EXTRACTION_PROMPT = """
Extract modernization-relevant facts from legacy backend source material.

Extract only facts that are directly supported by the input text.

Useful extraction classes:
- module
- technology
- database
- database_access_pattern
- external_dependency
- integration_protocol
- batch_job
- modernization_risk
- modernization_target
- configuration_value

For each extraction, add useful attributes such as:
- module
- risk_type
- severity
- reason
- suggested_modernization
- source_category

Use exact text spans from the input whenever possible.
Do not invent facts that are not supported by the text.
"""


# Few-shot examples teach LangExtract what shape of extraction we want.
# The extraction_text values are intentionally copied from the example text.
EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "BillingService uses JDBC Statement to execute SQL built by "
            "string concatenation against Oracle 11g."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="module",
                extraction_text="BillingService",
                attributes={
                    "source_category": "code",
                    "reason": "Service class name identifies the Billing module.",
                },
            ),
            lx.data.Extraction(
                extraction_class="database_access_pattern",
                extraction_text="JDBC Statement",
                attributes={
                    "risk_type": "raw_jdbc_statement",
                    "severity": "medium",
                    "suggested_modernization": "Use PreparedStatement, repository layer, or ORM.",
                },
            ),
            lx.data.Extraction(
                extraction_class="modernization_risk",
                extraction_text="string concatenation",
                attributes={
                    "risk_type": "sql_injection_risk",
                    "severity": "high",
                    "reason": "SQL built using string concatenation can be unsafe.",
                },
            ),
            lx.data.Extraction(
                extraction_class="database",
                extraction_text="Oracle 11g",
                attributes={
                    "risk_type": "outdated_database",
                    "severity": "high",
                    "suggested_modernization": "Assess migration to PostgreSQL or supported Oracle version.",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The Customer module sends address updates to an external CRM "
            "system using SOAP. The CRM call has no retry handling."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="module",
                extraction_text="Customer module",
                attributes={
                    "source_category": "documentation",
                },
            ),
            lx.data.Extraction(
                extraction_class="external_dependency",
                extraction_text="external CRM system",
                attributes={
                    "dependency_type": "external_system",
                    "reason": "The backend depends on a separate CRM system.",
                },
            ),
            lx.data.Extraction(
                extraction_class="integration_protocol",
                extraction_text="SOAP",
                attributes={
                    "risk_type": "legacy_integration_protocol",
                    "suggested_modernization": "Wrap or replace SOAP integration with REST API if feasible.",
                },
            ),
            lx.data.Extraction(
                extraction_class="modernization_risk",
                extraction_text="no retry handling",
                attributes={
                    "risk_type": "resilience_gap",
                    "severity": "medium",
                    "suggested_modernization": "Add timeout, retry, logging, and alerting.",
                },
            ),
        ],
    ),
]


# ---------------------------------------------------------------------
# 5. Utility: reset output directories
# ---------------------------------------------------------------------

def reset_outputs() -> None:
    """
    Deletes old vectorstore/output files so each run starts cleanly.

    Why?
    During experiments, repeated ingestion can create duplicate records.
    Clean reset keeps results easier to understand.
    """
   
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 6. Utility: read legacy backend files
# ---------------------------------------------------------------------

def load_legacy_files() -> List[Dict[str, str]]:
    """
    Reads supported files from legacy_backend.

    Returns a list like:
    [
        {
            "source_path": "code/billing/BillingService.java",
            "file_name": "BillingService.java",
            "extension": ".java",
            "text": "... file content ..."
        }
    ]
    """
    if not LEGACY_BACKEND_ROOT.exists():
        raise FileNotFoundError(
            f"Could not find legacy backend folder: {LEGACY_BACKEND_ROOT}"
        )

    loaded_files: List[Dict[str, str]] = []

    for path in LEGACY_BACKEND_ROOT.rglob("*"):
        # Skip folders.
        if not path.is_file():
            continue

        # Skip unsupported file types.
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8")

        # Relative path is better metadata than absolute Windows path.
        relative_path = path.relative_to(LEGACY_BACKEND_ROOT).as_posix()

        loaded_files.append(
            {
                "source_path": relative_path,
                "file_name": path.name,
                "extension": path.suffix,
                "text": text,
            }
        )

    return loaded_files


# ---------------------------------------------------------------------
# 7. Raw ingestion: original files/chunks
# ---------------------------------------------------------------------

def build_raw_documents(loaded_files: List[Dict[str, str]]) -> List[Document]:
    """
    Converts original files into LangChain Documents, then chunks them.

    These documents preserve the raw source material.

    Example user question this helps answer:
        "Show me where the SOAP call is made."
        "Which file contains JDBC Statement usage?"
    """

    full_file_documents: List[Document] = []

    for file_info in loaded_files:
        full_file_documents.append(
            Document(
                page_content=file_info["text"],
                metadata={
                    "record_type": "raw_chunk",
                    "source_path": file_info["source_path"],
                    "file_name": file_info["file_name"],
                    "extension": file_info["extension"],
                },
            )
        )

    # For source code and small docs, we use modest chunks.
    # overlap helps preserve context across chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    raw_chunks = splitter.split_documents(full_file_documents)

    # Add chunk index metadata so we can trace chunks later.
    for index, chunk in enumerate(raw_chunks):
        chunk.metadata["chunk_index"] = index

    return raw_chunks


# ---------------------------------------------------------------------
# 8. LangExtract ingestion: structured modernization records
# ---------------------------------------------------------------------

def run_langextract_for_file(file_info: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Runs LangExtract on one file and converts the extraction result into
    plain Python dictionaries.

    LangExtract returns extraction objects. We convert them to JSON-friendly
    dicts because those are easier to save, inspect, and vectorize.
    """

    source_path = file_info["source_path"]
    text = file_info["text"]

    result = lx.extract(
        text_or_documents=text,
        prompt_description=MODERNIZATION_EXTRACTION_PROMPT,
        examples=EXAMPLES,
        model_id=LANGEXTRACT_MODEL_ID,

        # For our tiny corpus, 1 pass is enough.
        # Later, for larger/messier docs, we could increase this.
        extraction_passes=1,

        # Smaller buffers can improve extraction precision.
        max_char_buffer=1200,
    )

    records: List[Dict[str, Any]] = []

    for extraction in result.extractions:
        # Important:
        # LangExtract may return char_interval=None if an extraction
        # cannot be grounded in the source text.
        # For our modernization knowledge base, we keep only grounded facts.
        if extraction.char_interval is None:
            continue

        record = {
            "record_type": "structured_modernization_record",
            "source_path": source_path,
            "file_name": file_info["file_name"],
            "extension": file_info["extension"],
            "extraction_class": extraction.extraction_class,
            "extraction_text": extraction.extraction_text,
            "attributes": extraction.attributes or {},
            "char_interval": {
                "start": extraction.char_interval.start_pos,
                "end": extraction.char_interval.end_pos,
            },
        }

        records.append(record)

    return records


def build_structured_records(
    loaded_files: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Runs LangExtract over all loaded files.
    """

    all_records: List[Dict[str, Any]] = []

    for file_info in loaded_files:
        print(f"Running LangExtract on: {file_info['source_path']}")

        records = run_langextract_for_file(file_info)
        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------
# 9. Convert structured records to vectorizable Documents
# ---------------------------------------------------------------------

def structured_record_to_text(record: Dict[str, Any]) -> str:
    """
    Converts one structured JSON record into readable text.

    Why?
    Vector databases embed text. So even though our record is structured,
    we create a natural-language representation for semantic search.
    """

    attributes_text = json.dumps(
        record["attributes"],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Structured modernization record

Source file: {record["source_path"]}
Extraction class: {record["extraction_class"]}
Extracted text: {record["extraction_text"]}

Attributes:
{attributes_text}
""".strip()


def build_structured_documents(
    records: List[Dict[str, Any]]
) -> List[Document]:
    """
    Converts LangExtract records into LangChain Documents.

    These documents form the second knowledge layer.
    """

    documents: List[Document] = []

    for index, record in enumerate(records):
        documents.append(
            Document(
                page_content=structured_record_to_text(record),
                metadata={
                    "record_type": "structured_modernization_record",
                    "record_index": index,
                    "source_path": record["source_path"],
                    "file_name": record["file_name"],
                    "extension": record["extension"],
                    "extraction_class": record["extraction_class"],
                    "extraction_text": record["extraction_text"],
                },
            )
        )

    return documents


# ---------------------------------------------------------------------
# 10. Save structured records to JSONL for inspection
# ---------------------------------------------------------------------

def save_records_jsonl(records: List[Dict[str, Any]]) -> None:
    """
    Saves extracted structured records to a JSONL file.

    JSONL = one JSON object per line.
    This is convenient for inspection and debugging.
    """

    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# 11. Store both layers in Chroma
# ---------------------------------------------------------------------

def ingest_into_chroma(
    raw_documents: List[Document],
    structured_documents: List[Document],
) -> None:
    """
    Stores both document types in the same Chroma persist directory,
    but in two different collections.

    Collection 1:
        legacy_backend_raw_chunks

    Collection 2:
        legacy_backend_structured_records

    This keeps the two layers separate but still under one legacy DB folder.
    """

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Raw source/code/docs collection.
    Chroma.from_documents(
        documents=raw_documents,
        embedding=embeddings,
        persist_directory=str(LEGACY_CHROMA_DIR),
        collection_name="legacy_backend_raw_chunks",
    )

    # Structured LangExtract facts collection.
    Chroma.from_documents(
        documents=structured_documents,
        embedding=embeddings,
        persist_directory=str(LEGACY_CHROMA_DIR),
        collection_name="legacy_backend_structured_records",
    )


# ---------------------------------------------------------------------
# 12. Main script
# ---------------------------------------------------------------------

def main() -> None:
    """
    Full ingestion flow:

    1. Reset previous experimental outputs.
    2. Read legacy_backend files.
    3. Create raw chunks.
    4. Run LangExtract to create structured modernization records.
    5. Save structured records to JSONL.
    6. Store raw chunks and structured records in Chroma.
    """

    print("Resetting old experimental outputs...")
    reset_outputs()

    print("Loading legacy backend files...")
    loaded_files = load_legacy_files()
    print(f"Loaded files: {len(loaded_files)}")

    print("Building raw source/code chunks...")
    raw_documents = build_raw_documents(loaded_files)
    print(f"Raw chunks created: {len(raw_documents)}")

    print("Running LangExtract and building structured records...")
    structured_records = build_structured_records(loaded_files)
    print(f"Structured records extracted: {len(structured_records)}")

    print("Saving structured records JSONL...")
    save_records_jsonl(structured_records)
    print(f"Saved: {OUTPUT_JSONL_PATH}")

    print("Building structured vector documents...")
    structured_documents = build_structured_documents(structured_records)
    print(f"Structured documents created: {len(structured_documents)}")

    print("Ingesting both layers into Chroma...")
    ingest_into_chroma(raw_documents, structured_documents)

    print("\nDone.")
    print(f"Chroma directory: {LEGACY_CHROMA_DIR}")
    print("Collections:")
    print("  - legacy_backend_raw_chunks")
    print("  - legacy_backend_structured_records")


if __name__ == "__main__":
    main()