"""
Brain Stakeholder Extractor
1. Selects a brain interactively (workspaces -> brains -> documents)
2. Fetches all documents from the selected brain
3. Extracts stakeholders (Name, Category, Role, Confidence Score)
4. Displays results in a clean table
"""

import asyncio
import json
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import re
from normalize_stakeholder import normalize_stakeholder_names
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Import from files
sys.path.insert(0, "supabase")  # Add supabase folder to Python path
from select_data import (
    select_brain_from_workspace,
    initialize_supabase,
    get_documents_per_brain,
)
from supabase_db import get_document_data, decode_string
# getdocumentdata reconstructs full doc text

# Stakeholder extraction schema
STAKEHOLDER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Stakeholder Name": {
                "type": "string",
                "description": "Exact name of the stakeholder/organization (e.g., 'Social Welfare Department')",
            },
            "Canonical Name": {
                "type": "string",
                "description": "Normalized name of the stakeholder/organization (e.g., 'Social Welfare Department' and 'SWD','Taiwanese Government' and 'Government of Taiwan')",
            },
            "Category": {
                "type": "string",
                "description": "Stakeholder category (Regulator, Supplier, Consumer, Competitor, Partner, etc.)",
            },
            "Role": {
                "type": "string",
                "description": "Specific role/description (e.g., 'Regulates public elderly services')",
            },
            "Confidence Score": {
                "type": "string",
                "description": "Confidence as percentage string (e.g., '95%', '90%', '85%')",
            },
            "Source metadata": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Source file name"},
                    "document_id": {
                        "type": "string",
                        "description": "Source document ID",
                    },
                    "chunk_index": {"type": "integer"},
                    "extraction_context": {
                        "type": "string",
                        "description": "Exact snippet containing stakeholder mention (max 300 chars)",
                    },
                },
                "required": ["filename", "extraction_context"],
            },
            "required": [
                "Stakeholder Name",
                "Category",
                "Role",
                "Confidence Score",
                "Source metadata",
            ],
        },
    },
}


def parse_json_response(raw_info: str) -> List[Dict]:
    """JSON parser"""
    if not raw_info:
        return []

    json_str = re.sub(r"```json?|\n*```", "", raw_info.strip())
    json_str = re.sub(r"^json\s*:?\s*", "", json_str, flags=re.IGNORECASE)

    try:
        parsed = json.loads(json_str)
        return [parsed] if isinstance(parsed, dict) else parsed
    # except:
    #     return []
    except Exception as e:
        print(f" Parse error: {type(e).__name__}: {e}")
        return []


async def extract_stakeholders_from_text(
    text: str, doc_id: str, filename: str, chunk_index: int = 0
) -> List[Dict[str, Any]]:
    """Extract stakeholders using enrichment pipeline."""
    # Create InputState and run the graph
    from enrichment.state import InputState
    from enrichment.configuration import Configuration
    from enrichment import graph

    source_header = (
        f"\n\n=== SOURCE METADATA ===\n"
        f"Document ID: {doc_id}\n"
        f"Filename: {filename}\n"
        f"Chunk Index: {chunk_index}\n"
        f"=== END METADATA ===\n\n"
    )

    full_input = source_header + text

    initial_state = InputState(topic=full_input, extraction_schema=STAKEHOLDER_SCHEMA)

    config = Configuration(
        model="openai/gpt-4o-mini",
        prompt=(
            "You are an assistant tasked with extracting specific information from the provided text using the extraction schema.\n\n"
            "Schema:\n{schema}\n\n"
            "Text:\n{topic}\n\n"
            "Please provide your answer directly in clear text, filling in the schema (In English)."
            "extraction_context in Source metadata: For each stakeholder, copy 200-300 chars around the mention from the document.\n"
            "Rules: "
            "Use only the information explicitly provided in the input text."
            "Do not add assumptions, external knowledge, or inferred entities not present in the text."
        ),
        max_loops=2,
    ).__dict__

    final_state = await graph.ainvoke(initial_state, config)

    # print(final_state.get("answer", ""))
    return parse_json_response(final_state.get("answer", ""))


def calculate_splitter_params(model_context=128000) -> tuple:
    # Use ~40-50% of context for chunk payload
    base_chunk = int(model_context * 0.50)
    CHARS_PER_TOKEN = 4  # English avg: 3.5-4.5 chars/token
    chunk_size = base_chunk * CHARS_PER_TOKEN

    # Overlap: 10-20% of chunk_size for semantic continuity
    overlap = int(chunk_size * 0.15)
    overlap = max(100, overlap)
    return chunk_size, overlap


async def extract_stakeholders_adaptive(text, doc_id, filename, threshold=100000):
    """Extract: Whole if small, chunk if large"""
    print(f"   {filename}: Text: {len(text) / 1000:.0f}k chars")

    if len(text) <= threshold:
        print("    → Whole doc extraction")
        return await extract_stakeholders_from_text(
            text=text, doc_id=doc_id, filename=filename, chunk_index=0
        )

    chunk_size, chunk_overlap = calculate_splitter_params()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # ~256k chars
        chunk_overlap=chunk_overlap,  # ~38k chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(
        f"    → {len(chunks)} chunks ({chunk_size // 1000}k/{chunk_overlap // 1000}k)"
    )

    all_stakeholders = []
    tasks = []  # CHANGE: Parallelize chunks

    for i, chunk in enumerate(chunks, 1):
        task = asyncio.create_task(
            extract_stakeholders_from_text(chunk, doc_id, filename, i)
        )
        tasks.append(task)

    try:
        results = await asyncio.gather(
            *tasks, return_exceptions=True
        )  # Gather with exceptions
        for result in results:
            if isinstance(result, Exception):
                print(f"    Chunk error: {type(result).__name__}: {result}")
            elif isinstance(result, list):
                all_stakeholders.extend(result)
    except Exception as e:
        print(f"    Error gathering chunk results: {type(e).__name__}: {e}")

    return all_stakeholders


async def extract_all_stakeholders_from_brain(brain_name: str, brain_id: str):
    """Extract ALL stakeholders from ALL documents"""

    print(f"Processing ALL documents from brain: {brain_id}")

    supabase = initialize_supabase()

    # Get all documents for this brain
    documents = get_documents_per_brain(supabase, brain_id)
    if not documents:
        print("No documents found in this brain.")
        return

    print(f"Found {len(documents)} documents. Fetching content...")

    ##################### REMOVE AFTER TESTING #####################
    FILTER_DOC_NUM = True
    if FILTER_DOC_NUM:
        print(f"Limiting to first 3 documents for testing.")
        documents = documents[:3]

    ###############################################################

    # Fetch full text for each document
    all_stakeholders = []
    for i, doc in enumerate(documents, 1):
        doc_id = doc["id"]
        filename = doc.get("file_name", "Unknown")
        print(f"  {i}/{len(documents)}: {filename} ({doc_id[:8]}...)")

        try:
            full_text = get_document_data(supabase, doc_id)
            full_text = full_text.encode("utf-8").decode("unicode_escape")
            if full_text:
                stakeholders = await extract_stakeholders_adaptive(
                    text=full_text, doc_id=doc_id, filename=filename
                )
                all_stakeholders.extend(stakeholders)
        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    all_stakeholders = await normalize_stakeholder_names(all_stakeholders)
    print(f"Normalized: {len(all_stakeholders)} stakeholders with canonical names.")

    # Save JSON
    output = {
        "brain": brain_name,
        "brain_id": brain_id,
        "total_stakeholders": len(all_stakeholders),
        "stakeholders": all_stakeholders,
    }
    return output


async def main():
    #  TEST MODE - Load from real TXT file
    test_file_path = Path("stakeholder_pipeline/test_policy_text.txt")
    if test_file_path.exists():
        print(" TEST MODE: Loading from test_policy_text.txt")

        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                test_text = f.read()

            mock_doc_id = "test_doc_001"
            mock_filename = test_file_path.name

            # Extract → exactly like real documents!
            stakeholders = await extract_stakeholders_adaptive(
                text=test_text, doc_id=mock_doc_id, filename=mock_filename
            )

            # Normalize
            stakeholders = await normalize_stakeholder_names(stakeholders)

            # Results
            output = {
                "brain": "Test Elderly Policy",
                "brain_id": "test_123",
                "test_file": str(test_file_path),
                "total_stakeholders": len(stakeholders),
                "stakeholders": stakeholders,
            }

            print(
                f"\n SUCCESS: Extracted {len(stakeholders)} stakeholders from {len(test_text) / 1000:.0f}k chars!"
            )
            print(json.dumps(output, indent=2, ensure_ascii=False)[:1000] + "...")

            # Save
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(
                output_dir / "test_policy_output.json", "w", encoding="utf-8"
            ) as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(" Results → output/test_policy_output.json")

            return

        except Exception as e:
            print(f" Test file error: {e}")

    print("ℹ  No test_data/elderly_policy_2026.txt → Running real brain extraction...")

    ################### Brain Stakeholder Extractor#############################
    print("Brain Stakeholder Extractor")
    print("=" * 50)

    # Use select() to pick brain
    selected_brain = select_brain_from_workspace()
    if not selected_brain:
        print("No brain selected.")
        return

    try:
        brain_id = selected_brain["brain_id"]
    except KeyError:
        print("Invalid brain data - missing brain_id")
        return
    brain_name = selected_brain.get("name", "Unknown")

    print(f"\nSelected Brain: {brain_name} (ID: {brain_id})")
    output = await extract_all_stakeholders_from_brain(brain_name, brain_id)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"stakeholders_output_{output['brain']}.json"
    with open(output_dir / output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n Full results saved to: {output_file}.json")


if __name__ == "__main__":
    asyncio.run(main())
