"""
Brain Stakeholder Extractor
1. Selects a brain interactively (workspaces -> brains -> documents)
2. Fetches all documents from the selected brain
3. Extracts stakeholders (Name, Category, Role, Confidence Score)
4. Displays results in a clean table
"""

import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use absolute package imports
from stakeholder_pipeline.normalize_stakeholder import normalize_stakeholder_names
from stakeholder_pipeline.utils import (
    save_output,
    calculate_splitter_params,
    parse_json_response,
    calculate_threshold,
)
from supabase_utils.select_data import (
    select_brain_from_workspace,
    initialize_supabase,
    get_documents_per_brain,
)
from supabase_utils.supabase_db import (
    get_document_data,
    decode_string,
)  # getdocumentdata reconstructs full doc text


load_dotenv()


# Define Project Root (Up one level from stakeholder_pipeline)
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stakeholder extraction schema
STAKEHOLDER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Stakeholder Name": {
                "type": "string",
                "description": "Extract the EXACT name of the stakeholder/organization as mentioned in the text.",
            },
            "Canonical Name": {
                "type": "string",
                "description": "Normalized the Stakeholder Name (e.g., 'Social Welfare Department' and 'SWD','Taiwanese Government' and 'Government of Taiwan') in English.",
            },
            "Category": {
                "type": "string",
                "description": "Stakeholder category (Regulator, Supplier, Consumer, Competitor, Partner, etc.)",
            },
            "Role": {
                "type": "string",
                "description": "Specific role/description of the entity(e.g., 'Regulates public elderly services')",
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
                    "evidence_original": {
                        "type": "string",
                        "description": "Copy the EXACT snippet from the text containing stakeholder mention (max 300 chars) as mentioned in the text.",
                    },
                    "evidence_translated": {
                        "type": "string",
                        "description": "English translation of evidence_original. English text stays English.",
                    },
                },
                "required": ["filename", "evidence_original", "evidence_translated"],
            },
            "required": [
                "Stakeholder Name",
                "Canonical Name",
                "Category",
                "Role",
                "Confidence Score",
                "Source metadata",
            ],
        },
    },
}


class StakeholderExtractor:
    MODEL_CONTEXTS = {
        "openai/gpt-4o-mini": 128000,
        "openai/gpt-4o": 128000,
    }

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_docs: int = None,
        output_dir: str = "output",
        concurrency_limit: int = 5,
    ):
        self.model = model
        self.max_docs = max_docs
        self.output_dir = output_dir
        self.concurrency_limit = concurrency_limit
        self.model_context = self.MODEL_CONTEXTS.get(model, 128000)
        self.threshold = calculate_threshold(self.model_context)
        self.supabase = initialize_supabase()
        self.max_loops = 2  # Limit loops
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s
        reraise=True,  # Re-raise final failure
    )
    async def extract_stakeholders_from_text(
        self, text: str, doc_id: str, filename: str, chunk_index: int = 0
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

        initial_state = InputState(
            topic=full_input, extraction_schema=STAKEHOLDER_SCHEMA
        )

        config = Configuration(
            model=self.model,
            prompt=(
                "You are an assistant tasked with extracting specific information from the provided text using the extraction schema.\n\n"
                "Schema:\n{schema}\n\n"
                "Text:\n{topic}\n\n"
                "Please provide your answer directly in clear text, filling in the schema."
                "Except for Stakeholder Name and evidence_original, rest of the schema should be filled in English."
                "Extract even if same entity is mentioned multiple times."
                "Rules: "
                "Use only the information explicitly provided in the input text."
                "Do not add assumptions, external knowledge, or inferred entities not present in the text."
            ),
            max_loops=self.max_loops,
        ).__dict__

        # "evidence_original in Source metadata: For each stakeholder, copy EXACT original text, 200-300 chars around the mention from the document.\n"
        # "evidence_translated in Source metadata: English translation of evidence_original.English text stays English.\n"
        # final_state = await graph.ainvoke(initial_state, config)

        # Use semaphore to prevent hitting rate limits during gather
        async with self.semaphore:
            print(f"    Processing {filename}...")
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config), timeout=300
            )  # 5 mins

        # print(final_state.get("answer", ""))
        return parse_json_response(final_state.get("answer", ""))

    async def extract_stakeholders_adaptive(self, text, doc_id, filename):
        """Extract: Whole if small, chunk if large"""
        # print(f"   {filename}: Text: {len(text) / 1000:.0f}k chars")

        if len(text) <= self.threshold:
            print(f"    → File:{filename} - Whole doc extraction.")
            return await self.extract_stakeholders_from_text(
                text=text, doc_id=doc_id, filename=filename, chunk_index=0
            )

        chunk_size, chunk_overlap = calculate_splitter_params(self.model_context)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # ~256k chars
            chunk_overlap=chunk_overlap,  # ~38k chars
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        print(
            f"    → File:{filename} - {len(chunks)} chunks ({chunk_size // 1000}k/{chunk_overlap // 1000}k)."
        )

        tasks = []  # CHANGE: Parallelize chunks

        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(
                self.extract_stakeholders_from_text(chunk, doc_id, filename, i)
            )
            tasks.append(task)

        try:
            results = await asyncio.gather(
                *tasks, return_exceptions=True
            )  # Gather with exceptions
        except Exception as e:
            print(f"    Error gathering chunk results: {type(e).__name__}: {e}")

        all_stakeholders = []
        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"    Chunk error: {type(result).__name__}: {result}")
            elif isinstance(result, list):
                all_stakeholders.extend(result)
                successful += 1
        print(f"   {successful}/{len(tasks)} chunks successful")

        return all_stakeholders

    async def extract_all_stakeholders_from_brain(self, brain_name: str, brain_id: str):
        """Extract ALL stakeholders from ALL documents"""

        print(f"Processing ALL documents from brain: {brain_id}")

        supabase = self.supabase

        # Get all documents for this brain
        documents = get_documents_per_brain(supabase, brain_id)
        if not documents:
            print("No documents found in this brain.")
            return []

        print(f"Found {len(documents)} documents. Fetching content...")

        if self.max_docs:
            documents = documents[: self.max_docs]
            print(f"Extracting from {self.max_docs} documents (FOR TESTING)")

        # Fetch full text for each document
        doc_tasks = []
        for i, doc in enumerate(documents, 1):
            doc_id = doc["id"]
            filename = doc.get("file_name", "Unknown")
            print(f"  {i}/{len(documents)}: {filename} ({doc_id[:8]}...)")

            try:
                raw_text = get_document_data(self.supabase, doc_id)
                full_text = decode_string(raw_text)
                # full_text = raw_text.encode("utf-8").decode("unicode_escape")
                task = self.extract_stakeholders_adaptive(
                    text=full_text, doc_id=doc_id, filename=filename
                )  # coroutine object of extract_stakeholders_adaptive for each doc
                doc_tasks.append(task)
            except Exception as e:
                print(f" Prep failed {filename}: {e}")
                doc_tasks.append(None)  # Skip bad doc

        results = await asyncio.gather(*doc_tasks, return_exceptions=True)

        # # Parallelize docs
        # doc_tasks = []
        # for i, doc in enumerate(documents, 1):
        #     doc_id = doc["id"]
        #     filename = doc.get("file_name", "Unknown")
        #     print(f"  {i}/{len(documents)}: {filename} ({doc_id[:8]}...)")

        #     async def process_doc(d=doc):  # Capture doc
        #         async with self.semaphore:
        #             try:
        #                 raw_text = get_document_data(self.supabase, d["id"])
        #                 full_text = decode_string(raw_text)
        #                 return await self.extract_stakeholders_adaptive(
        #                     full_text, d["id"], d.get("file_name", "Unknown")
        #                 )
        #             except Exception as e:
        #                 print(f"Failed {d['id']}: {e}")
        #                 return []

        #     doc_tasks.append(process_doc())

        # results = await asyncio.gather(*doc_tasks, return_exceptions=True)

        all_stakeholders = []
        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f" Doc failed: {result}")
            elif isinstance(result, list):
                all_stakeholders.extend(result)
                successful += 1

        print(f" {successful}/{len(doc_tasks)} docs complete")
        # all_stakeholders = await normalize_stakeholder_names(all_stakeholders)
        # print(f"Normalized: {len(all_stakeholders)} stakeholders with canonical names.")

        # Save JSON
        output = {
            "brain": brain_name,
            "brain_id": brain_id,
            "total_stakeholders": len(all_stakeholders),
            "stakeholders": all_stakeholders,
        }
        return output


async def run_test_mode():
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
            extractor = StakeholderExtractor(model="openai/gpt-4o")
            stakeholders = await extractor.extract_stakeholders_adaptive(
                text=test_text, doc_id=mock_doc_id, filename=mock_filename
            )
            # print(stakeholders)
            # Normalize
            # stakeholders = await normalize_stakeholder_names(stakeholders)

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


async def main():
    RUNNING_TEST_MODE = False  # CHANGE: Set to False to run real extraction
    if RUNNING_TEST_MODE:
        await run_test_mode()
        return

    #### Brain Stakeholder Extractor###
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

    import time

    start = time.time()
    extractor = StakeholderExtractor(
        model="openai/gpt-4o-mini", max_docs=3
    )  # CHANGE: Limit to 3 docs for testing

    result = await extractor.extract_all_stakeholders_from_brain(brain_name, brain_id)

    # Handle empty brain
    if not result or not result.get("stakeholders"):
        print(f"Brain '{brain_name}' has no documents or no stakeholders found.")
        result = {
            "brain": brain_name,
            "brain_id": brain_id,
            "stakeholders": [],
            "total_stakeholders": 0,
        }

    elapsed = time.time() - start

    output_dir = extractor.output_dir

    output_file = f"{result['brain']}_stakeholders.json"

    output_path = save_output(result, output_file, output_dir)
    print(f"\n Full results saved to: {output_path}")
    print(f"     Time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
