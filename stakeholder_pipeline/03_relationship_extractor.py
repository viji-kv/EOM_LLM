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
                        "description": "Canonical stakeholder name from the provided list",
                    },
                    "target": {
                        "type": "string",
                        "description": "Canonical stakeholder name from the provided list",
                    },
                    "relationship_description": {
                        "type": "string",
                        "description": "A short description of the type of relationship\n",
                    },
                    "relationship_category": {
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
                        "description": "High-level category",
                    },
                    "confidence": {"type": "string", "description": "90%, 85%, etc."},
                    "source_metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "document_id": {"type": "string"},
                            "evidence_original": {
                                "type": "string",
                                "description": "Copy 1 or 2 sentences AROUND each stakeholder mention in the original language.",
                            },
                            "evidence_translated": {
                                "type": "string",
                                "description": "evidence_original translated into English",
                            },
                        },
                        "required": [
                            "file_name",
                            "document_id",
                            "evidence_original",
                            "evidence_translated",
                        ],
                    },
                },
                "required": [
                    "source",
                    "target",
                    "relationship_description",
                    "relationship_category",
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
                        "description": "Canonical stakeholder name from the provided list",
                    },
                    "painpoint": {
                        "type": "string",
                        "description": "Specific problem/gaps/challenges ONLY, in one sentence.",
                    },
                    "painpoint_category": {
                        "type": "string",
                        "description": "A short category name (e.g.,Staff shortage, Cost, Healthcare Access, Financial Barriers)\n",
                        # "enum": [
                        #     "staffing",
                        #     "cost",
                        #     "compliance",
                        #     "access",
                        #     "quality",
                        #     "other",
                        # ],
                    },
                    "confidence": {"type": "string", "description": "90%, 85%, etc."},
                    "source_metadata": {  # Added for traceability
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "document_id": {"type": "string"},
                            "evidence_original": {
                                "type": "string",
                                "description": "Copy 1 or 2 sentences AROUND each painpoint mention in the original language.",
                            },
                            "evidence_translated": {
                                "type": "string",
                                "description": "evidence_original translated into English",
                            },
                        },
                        "required": [
                            "file_name",
                            "document_id",
                            "evidence_original",
                            "evidence_translated",
                        ],
                    },
                },
                "required": [
                    "stakeholder",
                    "painpoint",
                    "painpoint_category",
                    "source_metadata",
                ],
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
        model: str = "openai/gpt-4o",
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

    async def extract_alias(
        self,
        consolidated_data: dict,
    ) -> dict[str, list[str]]:
        """Extract all original names from all_sources to build alias mapping."""
        alias_map = {}

        for stakeholder in consolidated_data["consolidated_stakeholders"]:
            canonical = stakeholder["Canonical Name"]
            if not canonical:
                continue
            all_names = [canonical]  # Start with main name

            # Extract ALL original names from all_sources
            if (
                "consolidation_info" in stakeholder
                and "all_sources" in stakeholder["consolidation_info"]
            ):
                for source in stakeholder["consolidation_info"]["all_sources"]:
                    all_names.append(source["original_name"])

            # Deduplicate and filter
            unique_names = list(
                set([name.strip() for name in all_names if name.strip()])
            )
            # alias_map[canonical] = unique_names
            if canonical in alias_map:
                current = set(alias_map[canonical])
                alias_map[canonical].extend(
                    name for name in unique_names if name not in current
                )
            else:
                alias_map[canonical] = unique_names
        return alias_map

    # Added helper to chunk the stakeholder list into batches
    def get_stakeholder_batches(self, entities: List[str]):
        """Splits the stakeholder list into manageable batches."""
        for i in range(0, len(entities), self.MAX_ENTITIES_PER_PROMPT):
            yield entities[i : i + self.MAX_ENTITIES_PER_PROMPT]

    def format_entities_prompt(self, alias_map: dict[str, list[str]]) -> str:
        """Format canonical→aliases for LLM matching."""
        lines = ["ALLOWED STAKEHOLDERS (CANONICAL → ALIASES):"]
        for canonical, aliases in alias_map.items():
            # Show canonical | all surface forms found in docs
            display_names = [canonical] + aliases[1:]  # Skip duplicate canonical
            lines.append(
                f"CANONICAL NAME: {canonical} -> ALIAS: {' | '.join(display_names)}"
            )
        return "\n".join(lines)

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
        alias_map: dict[str, list[str]],
        doc_id: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Core extraction with batching and concurrency control."""
        from enrichment.state import InputState
        from enrichment.configuration import Configuration
        from enrichment import graph

        aggregated_results = {"relationships": [], "pain_points": []}
        canonical_stakeholders = list(alias_map.keys())

        # Loop through batches of stakeholders to ensure all are processed
        for batch in self.get_stakeholder_batches(canonical_stakeholders):
            batch_alias_map = {k: v for k, v in alias_map.items() if k in batch}
            entities_prompt = self.format_entities_prompt(
                batch_alias_map
            )  # canonical names
            source_header = f"SOURCE METADATA\nDocument ID: {doc_id}\nFilename: {filename}\nEND METADATA"

            # Refined rules for evidence and metadata persistence
            full_prompt = f"""{source_header}

    {entities_prompt}

    TEXT:
    {text}  

    RULES:

    RELATIONSHIP EXTRACTION:
    - Relationships: Extract interactions between members of the ALLOWED LIST. The format is: CANONICAL_NAME -> alias1 | alias2 | alias3
    - Extract all explicitly stated OR strongly implied relationships based on verbs and context
    - RELATIONSHIP_DESCRIPTION: Provide a brief (1-sentence) explanation of the interaction.
    - ALWAYS output the CANONICAL NAME (leftmost) in 'source', 'target', and 'stakeholder'
    

    PAIN POINT EXTRACTION (FRICTION & ATTRIBUTION):
    - Pain points: Extract issues explicitly tied to the stakeholders from ALLOWED LIST.
    - Only extract if the text shows a challenge, risk, grievance, or barrier. The evidence should show negative impact. 
    - Do not infer potential future problems or "implied" dependency. You MAY extract structural or systemic friction if the text clearly implies operational strain.
    - If the evidence snippet describes a successful partnership, a donation, or a standard activity without using words of struggle (e.g., 'insufficient', 'failing', 'difficult', 'declining'), you MUST NOT extract it as a pain point.
    - Ensure the 'Stakeholder' assigned to the pain point is the group actually experiencing the problem.
    - General statements of "Following the rules" are NOT pain points. Only extract if the compliance is described as a "burden," "difficulty","limitation.. etc"
    
    SOURCE METADATA:
    - evidence_original in Source metadata: For each stakeholder, copy 1 or 2 sentences around the mention from the document in its original language. If entities are far apart, use 'snippet...snippet'.\n"
    - evidence_translated in Source metadata: English translation of evidence_original.\n"
    - You MUST copy the Document ID and Filename into EVERY object.

"""
            # Only extract pain points explicitly stated as a challenge, risk, or grievance. Do not infer potential future problems. Also ensure that the 'Stakeholder' assigned to a pain point is the group actually experiencing the problem, not just a group mentioned in the same paragraph.
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
                print(f"   Processing {len(batch)} stakeholders for {filename}...")
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
        self, text: str, alias_map: dict[str, list[str]], doc_id: str, filename: str
    ) -> Dict[str, Any]:
        """Adaptive chunking"""
        # print(f"  {filename}: {len(text) / 1000:.1f}k chars")
        if len(text) <= self.threshold:
            print(f" -> File: {filename} - Whole document extraction.")
            result = await self.extract_from_chunk(text, alias_map, doc_id, filename)

            return result

        chunk_size, chunk_overlap = calculate_splitter_params(self.model_context)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        print(
            f"    → File:{filename} - {len(chunks)} chunks ({chunk_size // 1000}k/{chunk_overlap // 1000}k)."
        )

        tasks = [
            self.extract_from_chunk(c, alias_map, doc_id, filename) for c in chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

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
        self, brain_id: str, alias_map: dict[str, list[str]]
    ) -> Dict[str, Any]:
        """Full brain extraction (matching 01)."""
        documents = get_documents_per_brain(self.supabase, brain_id)

        print(f"Processing {len(documents)} docs in parallel...")

        if self.max_docs:
            documents = documents[: self.max_docs]
            print(f"Extracting from {self.max_docs} documents (FOR TESTING)")

        # Doc are sequential
        doc_tasks = []
        for i, doc in enumerate(documents, 1):
            doc_id = doc["id"]
            filename = doc.get("file_name", f"Doc_{i}")

            try:
                raw_text = get_document_data(self.supabase, doc_id)
                full_text = decode_string(raw_text)
                task = self.extract_adaptive(
                    full_text, alias_map, doc_id, filename
                )  # coroutine object of extract_adaptive for each doc
                doc_tasks.append(task)
            except Exception as e:
                print(f" Prep failed {filename}: {e}")
                doc_tasks.append(None)  # Skip bad doc

        # # Parallize docs
        # doc_tasks = []
        # for i, doc in enumerate(documents, 1):
        #     doc_id = doc["id"]
        #     filename = doc.get("file_name", f"Doc_{i}")
        #     print(f"  {i}/{len(documents)}: {filename}")

        #     async def process_doc(d=doc):
        #         async with self.semaphore:
        #             try:
        #                 raw_text = get_document_data(self.supabase, d["id"])
        #                 full_text = decode_string(raw_text)
        #                 return await self.extract_adaptive(
        #                     full_text,
        #                     canonical_stakeholders,
        #                     d["id"],
        #                     d.get("file_name", "Unknown"),
        #                 )  # coroutine object of extract_adaptive for each doc

        #             except Exception as e:
        #                 print(f" Prep failed {d['id']}: {e}")
        #                 return []

        #     doc_tasks.append(process_doc())
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

    async def canonical_names_from_file(self, input_file: str) -> dict[str, list[str]]:
        """Utility to load canonical names from a JSON file."""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        alias_map = await self.extract_alias(data)
        return alias_map


async def run_test_mode():
    #  TEST MODE - Load from real TXT file
    input_file = "output/test_policy_output_consolidated.json"
    test_file_path = Path("stakeholder_pipeline/test_policy_text.txt")
    if test_file_path.exists():
        print(" TEST MODE: Loading from test_policy_text.txt")

        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                test_text = f.read()

            mock_doc_id = "test_doc_001"
            mock_filename = test_file_path.name

            # test_alias_map = {
            #     "LTA": ["LTA", "Land Transport Authority"],
            #     "MOT": ["MOT", "Ministry of Transport"],
            #     "MOF": ["MOF", "Ministry of Finance"],
            #     "SMRT Corporation": ["SMRT Corporation", "SMRT"],
            #     "SBS Transit": ["SBS Transit"],
            #     "Grab Singapore": ["Grab Singapore", "Grab"],
            #     "Gojek Singapore": ["Gojek Singapore", "Gojek"],
            #     "NEA": ["NEA", "National Environment Agency"],
            #     "Town Councils": ["Town Councils"],
            #     "CAS": [
            #         "CAS",
            #         "Commuters’ Association of Singapore",
            #         "Commuters Association of Singapore",
            #     ],
            #     "Public Transport Operators": ["Public Transport Operators"],
            #     "Technology Firms": ["Technology Firms"],
            #     "Private Mobility Providers": ["Private Mobility Providers"],
            #     "Fleet Operators": ["Fleet Operators"],
            # }

            extractor = RelationshipExtractor(
                output_dir="output", concurrency_limit=3, max_docs=3
            )
            test_alias_map = await extractor.canonical_names_from_file(input_file)
            result = await extractor.extract_adaptive(
                test_text, test_alias_map, mock_doc_id, mock_filename
            )

            print(
                f"\n SUCCESS: Extracted {len(result['relationships'])} relationships from {len(test_text) / 1000:.0f}k chars!"
            )
            print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...")

            # Save
            output_filename = "test_policy_output_relationship.json"

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

    import time

    start = time.time()

    extractor = RelationshipExtractor(
        output_dir="output", concurrency_limit=3, max_docs=3
    )

    alias_map = await extractor.canonical_names_from_file(input_file)

    result = await extractor.extract_from_brain(
        "ce88cb71-2528-4598-a585-5fa2dfd03319", alias_map
    )

    elapsed = time.time() - start

    input_path = Path(input_file).name
    output_filename = input_path.replace("_consolidated.json", "_relationships.json")

    output_path = save_output(
        result=result, output_filename=output_filename, output_dir=extractor.output_dir
    )
    print(
        f"Extracted {result['total_relationships']} rels + {result['total_pain_points']} pain points"
    )
    print(f"     Time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
