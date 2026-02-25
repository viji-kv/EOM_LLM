import asyncio
import json
import re
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Set
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.extraction_utils import (
    parse_json_response,
    calculate_splitter_params,
    save_output,
)
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Import from files
sys.path.insert(0, "supabase")  # Add supabase folder to Python path
from select_data import (
    select_brain_from_workspace,
    initialize_supabase,
    get_documents_per_brain,
)
from supabase_db import get_document_data, decode_string

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
                        "description": "200-300 chars evidence quote",
                    },
                    "confidence": {"type": "string", "description": "90, 85, etc."},
                    "source_metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "document_id": {"type": "string"},
                            "chunk_index": {"type": "integer"},
                        },
                        "required": ["file_name", "document_id", "chunk_index"],
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
                            "chunk_index": {"type": "integer"},
                        },
                        "required": ["file_name", "document_id", "chunk_index"],
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
    MAX_ENTITIES_PER_PROMPT = 25  # Prevent context overflow

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        threshold: int = 100000,
        max_docs: int = None,
        output_dir: str = "output",
        concurrency_limit: int = 5,
    ):
        self.model = model
        self.threshold = threshold
        self.max_docs = max_docs
        self.output_dir = output_dir
        self.concurrency_limit = concurrency_limit
        self.model_context = self.MODEL_CONTEXTS.get(model, 128000)
        self.supabase = initialize_supabase()
        self.max_loops = 2
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

    # 🔴 CHANGE: Added helper to chunk the stakeholder list into batches
    def _get_stakeholder_batches(self, entities: List[str]):
        """Splits the stakeholder list into manageable batches."""
        for i in range(0, len(entities), self.MAX_ENTITIES_PER_PROMPT):
            yield entities[i : i + self.MAX_ENTITIES_PER_PROMPT]

    def _format_entities_prompt(self, canonical_stakeholders: List[str]) -> str:
        """Truncate + format canonical list."""
        entities = canonical_stakeholders[: self.MAX_ENTITIES_PER_PROMPT]
        return "ALLOWED STAKEHOLDERS (exact canonical names):\n" + "\n".join(
            f"- {e}" for e in entities
        )

    # 🔴 CHANGE: Added a more robust local parser to handle markdown and text filler
    def _robust_json_parser(self, text: str) -> Dict[str, Any]:
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
            print(f"   × Parsing Error: {e}")
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
        chunk_index: int,
    ) -> Dict[str, Any]:
        """Core extraction with batching and concurrency control."""
        from enrichment.state import InputState
        from enrichment.configuration import Configuration
        from enrichment import graph

        aggregated_results = {"relationships": [], "pain_points": []}

        # 🔴 CHANGE: Loop through batches of stakeholders to ensure all are processed
        for batch in self._get_stakeholder_batches(canonical_stakeholders):
            entities_prompt = self._format_entities_prompt(batch)
            source_header = f"SOURCE METADATA\nDocument ID: {doc_id}\nFilename: {filename}\nChunk Index: {chunk_index}\nEND METADATA"

            # 🔴 CHANGE: Refined rules for evidence and metadata persistence
            full_prompt = f"""{source_header}

    {entities_prompt}

    TEXT:
    {text}  

    RULES:
    1. ONLY use stakeholders from ALLOWED list above.
    2. Relationships: Extract interactions between members of the list.
    3. Pain points: Extract issues explicitly tied to these stakeholders.
    4. Evidence: Translate the relevant evidence quote into English. If entities are far apart, use 'snippet...snippet'.
    5. Metadata: You MUST copy the Document ID, Filename, and Chunk Index into EVERY object.

    SCHEMA:
    {json.dumps(RELATIONSHIP_SCHEMA, indent=2)}"""

            initial_state = InputState(
                topic=full_prompt, extraction_schema=RELATIONSHIP_SCHEMA
            )
            config = Configuration(
                model=self.model,
                prompt=(
                    "You are an expert analyst. Extract information exactly as per the schema provided below.\n\n"
                    "Schema:\n{schema}\n\n"
                    "Data to Analyze:\n{topic}\n\n"  # This is where your 'full_prompt' gets injected
                    "Evidence must be a direct quote. Source metadata must be preserved for every item."
                    "Return ONLY valid JSON."
                ),
                max_loops=self.max_loops,
            ).__dict__

            # 🔴 CHANGE: Use semaphore to prevent hitting rate limits during gather
            async with self.semaphore:
                print(
                    f"   [Batch] Processing {len(batch)} stakeholders for {filename} (Chunk {chunk_index})..."
                )
                final_state = await graph.ainvoke(initial_state, config)

            raw_result = final_state.get("answer")
            # result = parse_json_response(raw_result)
            # 🔴 CHANGE: Use the robust parser instead of the utility one
            result = self._robust_json_parser(raw_result)

            if isinstance(result, dict):
                aggregated_results["relationships"].extend(
                    result.get("relationships", [])
                )
                aggregated_results["pain_points"].extend(result.get("pain_points", []))

        return aggregated_results

    async def extract_adaptive(
        self, text: str, canonical_stakeholders: List[str], doc_id: str, filename: str
    ) -> Dict[str, Any]:
        """Adaptive chunking (matching 01)."""
        print(f"  {filename}: {len(text) / 1000:.1f}k chars")
        if len(text) <= self.threshold:
            return await self.extract_from_chunk(
                text, canonical_stakeholders, doc_id, filename, 0
            )

        chunk_size, chunk_overlap = calculate_splitter_params(self.model_context)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)

        tasks = [
            self.extract_from_chunk(c, canonical_stakeholders, doc_id, filename, i)
            for i, c in enumerate(chunks)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        aggregated = {"relationships": [], "pain_points": []}
        # for result in results:
        #     if isinstance(result, dict):
        #         aggregated["relationships"].extend(result.get("relationships", []))
        #         aggregated["pain_points"].extend(result.get("pain_points", []))

        successful = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"  ❌ Chunk failed: {result}")
            elif isinstance(result, dict):
                aggregated["relationships"].extend(result.get("relationships", []))
                aggregated["pain_points"].extend(result.get("pain_points", []))
                successful += 1
        print(f"  ✅ {successful}/{len(tasks)} chunks successful")

        return aggregated

    async def extract_from_brain(
        self, brain_id: str, canonical_stakeholders: List[str]
    ) -> Dict[str, Any]:
        """Full brain extraction (matching 01)."""
        documents = get_documents_per_brain(self.supabase, brain_id)
        if self.max_docs:
            documents = documents[: self.max_docs]

        all_rels, all_pps = [], []
        for i, doc in enumerate(documents, 1):
            doc_id = doc["id"]
            filename = doc.get("file_name", "Unknown")
            print(f"  {i}/{len(documents)}: {filename}")

            try:
                full_text = get_document_data(self.supabase, doc_id)
                full_text = full_text.encode("utf-8").decode("unicode_escape")
                result = await self.extract_adaptive(
                    full_text, canonical_stakeholders, doc_id, filename
                )
                all_rels.extend(result["relationships"])
                all_pps.extend(result["pain_points"])
            except Exception as e:
                print(f"    ERROR {filename}: {e}")

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
            f"🔄 Deduplicated: {len(canonical_stakeholders)} → {len(unique)} stakeholders"
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


# Usage example (matching your pipeline)
async def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py stakeholders_output.json")
        sys.exit(1)

    input_file = sys.argv[1]

    extractor = RelationshipExtractor(
        output_dir="output", concurrency_limit=3, max_docs=3
    )

    canonicals = await extractor.canonical_names_from_file(input_file)

    # canonicals = [
    #     "Social Welfare Department",
    #     "Hong Kong Judiciary",
    #     "HSBC",
    # ]  # From consolidated stakeholders
    result = await extractor.extract_from_brain(
        "87c6d2ac-7e1f-4a2f-8d6e-e00e2b14d90f", canonicals
    )

    # output_dir = Path(extractor.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # input_file = Path(input_file).name
    # output_filename = input_file.replace(".json", "_relationship.json")
    # with open(output_dir / output_filename, "w") as f:
    #     json.dump(result, f, indent=2)

    input_path = Path(input_file).name
    # output_filename = f"{input_path.stem}_relationships.json"
    output_filename = input_path.replace(".json", "_relationships.json")

    output_path = save_output(
        result=result, output_filename=output_filename, output_dir=extractor.output_dir
    )
    print(
        f"Extracted {result['total_relationships']} rels + {result['total_pain_points']} pain points"
    )


if __name__ == "__main__":
    asyncio.run(main())

############################
# import json

# input_file = (
#     "output/stakeholders_output_AP HK - Financial industry_llm_only_consolidated.json"
# )
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# canonical_stakeholders = [
#     stakeholder["Canonical Name"] for stakeholder in data["consolidated_stakeholders"]
# ]

# len(canonical_stakeholders)


# print(deduplicate_canonical_stakeholders(canonical_stakeholders))
