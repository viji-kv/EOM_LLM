import asyncio
import json
import re
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Set
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from stakeholder_pipeline.utils import (
    # parse_json_response,
    calculate_splitter_params,
    save_output,
    calculate_threshold,
)
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Import from files
from supabase_utils.select_data import (
    initialize_supabase,
    get_documents_per_brain,
)
from supabase_utils.supabase_db import get_document_data, decode_string

RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Canonical stakeholder name",
                    },
                    "target": {
                        "type": "string",
                        "description": "Canonical stakeholder name",
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "regulates",
                            "funds",
                            "partners",
                            "supplies",
                            "competes",
                            "collaborates",
                            "other",
                        ],
                    },
                    "evidence": {
                        "type": "string",
                        "description": "200-300 chars AROUND each stakeholder mention.",
                    },
                    "confidence": {"type": "string", "description": "90%, 85%, etc."},
                    "source_metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "document_id": {"type": "string"},
                        },
                        "required": ["file_name", "document_id"],
                    },
                },
                "required": [
                    "source",
                    "target",
                    "relationship_type",
                    "source_metadata",
                ],
            },
        },
        "pain_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stakeholder": {
                        "type": "string",
                        "description": "Canonical stakeholder",
                    },
                    "pain_point": {"type": "string"},
                    "pain_type": {
                        "type": "string",
                        "enum": [
                            "staffing",
                            "cost",
                            "compliance",
                            "access",
                            "quality",
                            "other",
                        ],
                    },
                    "evidence": {
                        "type": "string",
                        "description": "200-300 chars evidence quote",
                    },
                    "confidence": {"type": "string", "description": "90, 85, etc."},
                    "source_metadata": {  # Added for traceability
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "document_id": {"type": "string"},
                        },
                        "required": ["file_name", "document_id"],
                    },
                },
                "required": ["stakeholder", "pain_point", "source_metadata"],
            },
        },
    },
    "required": ["relationships", "pain_points"],
}


class RelationshipExtractor:
    MODEL_CONTEXTS = {"openai/gpt-4o-mini": 128000, "openai/gpt-4o": 128000}
    MAX_ENTITIES_PER_PROMPT = 100  # Prevent context overflow

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
        self.max_loops = 2
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

    # def calculate_threshold(self):
    #     """45-50% of context for text (rest = prompt/schema overhead)"""
    #     chars_per_token = 4  # English avg
    #     return int(self.model_context * 0.5 * chars_per_token)  # ~230k for gpt-4o-mini

    # Added helper to chunk the stakeholder list into batches
    def get_stakeholder_batches(self, entities: List[str]):
        """Splits the stakeholder list into manageable batches."""
        for i in range(0, len(entities), self.MAX_ENTITIES_PER_PROMPT):
            yield entities[i : i + self.MAX_ENTITIES_PER_PROMPT]

    def format_entities_prompt(self, canonical_stakeholders: List[str]) -> str:
        """Truncate + format canonical list."""
        entities = canonical_stakeholders[: self.MAX_ENTITIES_PER_PROMPT]
        return "ALLOWED STAKEHOLDERS:\n" + "\n".join(f"- {e}" for e in entities)

    # Added a more robust local parser to handle markdown and text filler
    def robust_json_parser(self, text: str) -> Dict[str, Any]:
        """Cleans LLM response and extracts the JSON block safely."""
        try:
            # Look for content between first { and last }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                clean_json = match.group(1)
                return json.loads(clean_json)
            # Fallback for raw strings
            return json.loads(text)
        except Exception as e:
            print(f"    Parsing Error: {e}")
            return {"relationships": [], "pain_points": []}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s
        reraise=True,  # Re-raise final failure
    )
    async def extract_from_chunk(
        self,
        text: str,
        canonical_stakeholders: List[str],
        doc_id: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Core extraction with batching and concurrency control."""
        from enrichment.state import InputState
        from enrichment.configuration import Configuration
        from enrichment import graph

        aggregated_results = {"relationships": [], "pain_points": []}

        # Loop through batches of stakeholders to ensure all are processed
        for batch in self.get_stakeholder_batches(canonical_stakeholders):
            entities_prompt = self.format_entities_prompt(batch)  # canonical names
            source_header = f"SOURCE METADATA\nDocument ID: {doc_id}\nFilename: {filename}\nEND METADATA"

            # Refined rules for evidence and metadata persistence
            full_prompt = f"""{source_header}

    {entities_prompt}

    TEXT:
    {text}  

    RULES:
    1. ONLY use stakeholders from ALLOWED list above.
    2. Relationships: Extract interactions between members of the list. Extract all explicitly stated OR strongly implied relationships based on verbs and context and explain briefly.
    3. Pain points: Extract issues explicitly tied to these stakeholders.
    4. Evidence: Translate the relevant evidence quote into English. If entities are far apart, use 'snippet...snippet'.
    5. Metadata: You MUST copy the Document ID and Filename into EVERY object.

"""

            initial_state = InputState(
                topic=full_prompt, extraction_schema=RELATIONSHIP_SCHEMA
            )
            config = Configuration(
                model=self.model,
                prompt=(
                    "You are an expert analyst. Extract information exactly as per the schema provided below.\n\n"
                    "Schema:\n{schema}\n\n"
                    "Data to Analyze:\n{topic}\n\n"  #'full_prompt' gets injected
                    "Evidence must be a direct quote. Source metadata must be preserved for every item."
                    "Return ONLY valid JSON."
                ),
                max_loops=self.max_loops,
            ).__dict__

            # Use semaphore to prevent hitting rate limits during gather
            async with self.semaphore:
                print(
                    f"   [Batch] Processing {len(batch)} stakeholders for {filename}..."
                )
                final_state = await asyncio.wait_for(
                    graph.ainvoke(initial_state, config), timeout=300
                )

            raw_result = final_state.get("answer")
            # result = parse_json_response(raw_result)
            # Use the robust parser instead of the utility one
            result = self.robust_json_parser(raw_result)

            if isinstance(result, dict):
                aggregated_results["relationships"].extend(
                    result.get("relationships", [])
                )
                aggregated_results["pain_points"].extend(result.get("pain_points", []))

        return aggregated_results

    async def extract_adaptive(
        self, text: str, canonical_stakeholders: List[str], doc_id: str, filename: str
    ) -> Dict[str, Any]:
        """Adaptive chunking"""
        print(f"  {filename}: {len(text) / 1000:.1f}k chars")
        if len(text) <= self.threshold:
            result = await self.extract_from_chunk(
                text, canonical_stakeholders, doc_id, filename
            )
            print(
                f"Relationship result from whole doc:{len(result['relationships'])}"
            )  ############################
            return result

        chunk_size, chunk_overlap = calculate_splitter_params(self.model_context)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)

        tasks = [
            self.extract_from_chunk(c, canonical_stakeholders, doc_id, filename)
            for c in chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_relationships = sum(
            len(r["relationships"]) for r in results if isinstance(r, dict)
        )  ############################
        print(
            f"Total relationships across all chunks: {total_relationships}"
        )  ##############
        aggregated = {"relationships": [], "pain_points": []}

        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"   Chunk failed: {result}")
            elif isinstance(result, dict):
                aggregated["relationships"].extend(result.get("relationships", []))
                aggregated["pain_points"].extend(result.get("pain_points", []))
                successful += 1
        print(f"   {successful}/{len(tasks)} chunks successful")

        return aggregated

    async def extract_from_brain(
        self, brain_id: str, canonical_stakeholders: List[str]
    ) -> Dict[str, Any]:
        """Full brain extraction (matching 01)."""
        documents = get_documents_per_brain(self.supabase, brain_id)

        print(f"Processing {len(documents)} docs in parallel...")
        doc_tasks = []

        for i, doc in enumerate(documents[: self.max_docs or len(documents)]):
            doc_id = doc["id"]
            filename = doc.get("file_name", f"Doc_{i}")

            try:
                raw_text = get_document_data(self.supabase, doc_id)
                full_text = decode_string(raw_text)
                task = self.extract_adaptive(
                    full_text, canonical_stakeholders, doc_id, filename
                )  # coroutine object of extract_adaptive for each doc
                doc_tasks.append(task)
            except Exception as e:
                print(f" Prep failed {filename}: {e}")
                doc_tasks.append(None)  # Skip bad doc

        results = await asyncio.gather(*doc_tasks, return_exceptions=True)
        all_rels, all_pps = [], []
        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f" Doc failed: {result}")
            elif isinstance(result, dict):
                all_rels.extend(result["relationships"])
                all_pps.extend(result["pain_points"])
                successful += 1

        print(f" {successful}/{len(doc_tasks)} docs complete")

        return {
            "brain_id": brain_id,
            "total_relationships": len(all_rels),
            "total_pain_points": len(all_pps),
            "relationships": all_rels,
            "pain_points": all_pps,
        }

    async def deduplicate_canonical_stakeholders(
        self,
        canonical_stakeholders: List[str],
    ) -> List[str]:
        """
        Deduplicate canonical stakeholders while preserving order.

        """
        seen = set()
        unique = []

        for stakeholder in canonical_stakeholders:
            if stakeholder and stakeholder.strip():  # Skip empty
                cleaned = stakeholder.strip()
                if cleaned not in seen:
                    seen.add(cleaned)
                    unique.append(cleaned)

        print(
            f" Deduplicated: {len(canonical_stakeholders)} → {len(unique)} stakeholders"
        )
        if len(canonical_stakeholders) > len(unique):
            print(f"   Removed {len(canonical_stakeholders) - len(unique)} duplicates")

        return unique

    async def canonical_names_from_file(self, input_file: str) -> List[str]:
        """Utility to load canonical names from a JSON file."""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        canonicals = [
            stakeholder["Canonical Name"]
            for stakeholder in data["consolidated_stakeholders"]
        ]
        canonicals = await self.deduplicate_canonical_stakeholders(canonicals)
        return canonicals


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

            test_canonicals = [
                "SWD",
                "HA",
                "Labour and Welfare Bureau",
                "HKCSS",
                "Caritas Hong Kong",
                "Baptist Oi Kwan Social Service",
                "Residential care homes",
                "For-Profit Hospital Groups",
                "Family Caregivers",
                "Elderly Citizens",
                "District Elderly Community Centres",
            ]

            extractor = RelationshipExtractor(
                output_dir="output", concurrency_limit=3, max_docs=3
            )

            result = await extractor.extract_adaptive(
                test_text, test_canonicals, mock_doc_id, mock_filename
            )

            print(
                f"\n SUCCESS: Extracted {len(result['relationships'])} relationships from {len(test_text) / 1000:.0f}k chars!"
            )
            print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...")

            # Save
            output_filename = "test_policy_output.json"

            output_path = save_output(
                result=result,
                output_filename=output_filename,
                output_dir=extractor.output_dir,
            )

            return

        except Exception as e:
            print(f" Test file error: {e}")

    print("ℹ  No test_data/elderly_policy_2026.txt → Running real brain extraction...")


# Usage example (matching your pipeline)
async def main():
    RUNNING_TEST_MODE = True  # CHANGE: Set to False to run real extraction
    if RUNNING_TEST_MODE:
        await run_test_mode()
        return

    if len(sys.argv) != 2:
        print("Usage: python script.py stakeholders_output.json")
        sys.exit(1)

    input_file = sys.argv[1]

    extractor = RelationshipExtractor(
        output_dir="output", concurrency_limit=3, max_docs=3
    )

    canonicals = await extractor.canonical_names_from_file(input_file)

    result = await extractor.extract_from_brain(
        "ce88cb71-2528-4598-a585-5fa2dfd03319", canonicals
    )

    input_path = Path(input_file).name
    output_filename = input_path.replace(".json", "_relationships.json")

    output_path = save_output(
        result=result, output_filename=output_filename, output_dir=extractor.output_dir
    )
    print(
        f"Extracted {result['total_relationships']} rels + {result['total_pain_points']} pain points"
    )


if __name__ == "__main__":
    asyncio.run(main())
