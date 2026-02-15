#!/usr/bin/env python3
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
        },
        "required": ["Stakeholder Name", "Category", "Role", "Confidence Score"],
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
    except:
        return []


async def extract_stakeholders_from_text(supabase, text: str) -> List[Dict[str, Any]]:
    """Extract stakeholders using enrichment pipeline."""
    # Create InputState and run the graph
    from enrichment.state import InputState
    from enrichment.configuration import Configuration
    from enrichment import graph

    initial_state = InputState(topic=text, extraction_schema=STAKEHOLDER_SCHEMA)

    config = Configuration(
        model="openai/gpt-4o-mini",
        prompt=(
            "You are an assistant tasked with extracting specific information from the provided text using the extraction schema.\n\n"
            "Schema:\n{schema}\n\n"
            "Text:\n{topic}\n\n"
            "Please provide your answer directly in clear text, filling in the schema (In English)."
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


async def extract_stakeholders_adaptive(supabase, full_text, threshold=100000):
    """Extract: Whole if small, chunk if large"""
    print(f"    Text: {len(full_text) / 1000:.0f}k chars")

    if len(full_text) <= threshold:
        print("    → Whole doc extraction")
        return await extract_stakeholders_from_text(supabase, full_text)

    chunk_size, chunk_overlap = calculate_splitter_params()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # ~256k chars
        chunk_overlap=chunk_overlap,  # ~38k chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    print(
        f"    → {len(chunks)} chunks ({chunk_size // 1000}k/{chunk_overlap // 1000}k)"
    )

    all_stakeholders = []

    for i, chunk in enumerate(chunks, 1):
        print(f"    Chunk {i}/{len(chunks)}")
        try:
            stakeholders = await extract_stakeholders_from_text(supabase, chunk)
            all_stakeholders.extend(stakeholders)
        except Exception as e:
            print(f"    Chunk {i} error: {e}")
            continue

    return all_stakeholders


FILTER_DOC_NUM = True


async def main():
    supabase = initialize_supabase()

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

    # Get all documents for this brain
    documents = get_documents_per_brain(supabase, brain_id)
    if not documents:
        print("No documents found in this brain.")
        return

    print(f"Found {len(documents)} documents. Fetching content...")

    if FILTER_DOC_NUM:
        print(f"Limiting to first 3 documents for testing.")
        documents = documents[:3]

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
                # stakeholders = await extract_stakeholders_from_text(supabase, full_text)
                stakeholders = await extract_stakeholders_adaptive(supabase, full_text)
                all_stakeholders.extend(stakeholders)
        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    all_stakeholders = await normalize_stakeholder_names(all_stakeholders)
    print(f"Normalized: {len(all_stakeholders)} stakeholders with canonical names.")

    # Save JSON
    output = {
        "brain": brain_name,
        "brain_id": brain_id,
        "stakeholders": all_stakeholders,
    }
    # print(f"OUTPUT:{output}")
    output_file = f"stakeholders_output_{output['brain']}"
    with open(output_file + ".json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n Full results saved to: {output_file}.json")


if __name__ == "__main__":
    asyncio.run(main())


# # # #######################################

# from enrichment.state import InputState
# from enrichment.configuration import Configuration
# from enrichment import graph


# # initial_state = InputState(topic=text, extraction_schema=STAKEHOLDER_SCHEMA)

# config = Configuration(
#     model="openai/gpt-4o-mini",
#     prompt=(
#         "You are an assistant tasked with extracting specific information from the provided text using the extraction schema.\n\n"
#         "Schema:\n{schema}\n\n"
#         "Text:\n{topic}\n\n"
#         "Please provide your answer directly in clear text, filling in the schema (In English)."
#         "CRITICAL: Multiple stakeholders expected. Return **COMPLETE ARRAY**.\n"
#         "Even if only 1 stakeholder found, use array format: [{}]\n\n"
#         # "RAW JSON ARRAY ONLY. NO TEXT. NO EXPLANATION. NO MARKDOWN.\n\n"
#     ),
#     max_loops=2,
# ).__dict__

# supabase = initialize_supabase()
# # selections = select_brain_from_workspace()
# brain_id = "0007f8b8-72d9-4952-b0a3-0fd2ff1cc2ed"
# # selected_brain = selections["brains"]
# documents = get_documents_per_brain(supabase, brain_id)
# print(f"Found {len(documents)} documents")
# documents[0]
# full_text = get_document_data(supabase, '0e677b8e-f337-427c-83b8-9e06ac991bd2')
# # stakeholders = await extract_stakeholders_adaptive(supabase, full_text)

# # Fetch full text for each document
# all_stakeholders = []
# for i, doc in enumerate(documents, 1):
#     doc_id = doc["id"]
#     filename = doc.get("file_name", "Unknown")
#     print(f"  {i}/{len(documents)}: {filename} ({doc_id[:8]}...)")

#     try:
#         full_text = get_document_data(supabase, doc_id)
#         full_text = full_text.encode("utf-8").decode("unicode_escape")
#         if full_text:
#             # stakeholders = await extract_stakeholders_from_text(supabase, full_text)
#             stakeholders = await extract_stakeholders_adaptive(supabase, full_text)
#             all_stakeholders.extend(stakeholders)
#     except Exception as e:
#         print(f"    Error processing {filename}: {e}")

# all_stakeholders = await normalize_stakeholder_names(all_stakeholders)
# print(f"Normalized: {len(all_stakeholders)} stakeholders with canonical names.")

# # Save JSON
# output = {
#     "brain": brain_name,
#     "brain_id": brain_id,
#     "stakeholders": all_stakeholders,
# }
# # print(f"OUTPUT:{output}")
