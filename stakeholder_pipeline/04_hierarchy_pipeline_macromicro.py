"""Stakeholder Hierarchy Pipeline"""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from stakeholder_pipeline.transform_stakeholders import transform_stakeholder_data
from stakeholder_pipeline.utils import save_output

# BATCH CONFIG
BATCH_SIZE = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hierarchy.log")],
)
logger = logging.getLogger(__name__)

# ===== EXISTING IMPORTS =====
from dotenv import load_dotenv

from enrichment import graph
from enrichment.configuration import Configuration
from enrichment.state import InputState
from stakeholder_pipeline.utils import parse_json_response

load_dotenv()


@dataclass
class HierarchyConfig:
    model: str = "openai/gpt-4o-mini"
    max_loops: int = 2
    output_dir: str = "output"
    # Batch settings
    batch_size: int = BATCH_SIZE


HIERARCHY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "stakeholder": {"type": "string", "description": "Exact canonical_name"},
            "primary_level": {
                "type": "string",
                "enum": ["macro", "meso", "micro"],
                "description": "Primary ecosystem level (macro=policy makers, meso=executors, micro=end-users)",
            },
            "parent": {
                "type": ["string", "null"],
                "description": "Supervising entity (null for macro roots). Use exact canonical_name.",
            },
            "reasoning": {"type": "string", "maxLength": 150},
        },
        "required": ["stakeholder", "primary_level", "parent"],
    },
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def safe_llm_invoke(initial_state: InputState, llm_config: Dict) -> Dict:
    """Retry logic."""
    try:
        return await graph.ainvoke(initial_state, llm_config)
    except Exception as e:
        logger.warning(f"LLM retry {e}")
        raise


# SUMMARY BUILDING
def build_stakeholder_summaries(stakeholders: List[Dict]) -> List[str]:
    """Build summaries."""
    summaries = []
    for i, s in enumerate(stakeholders):
        canonical = s["canonical_name"]

        # Roles
        roles_str = "; ".join(
            [f"{cat}- {role}" for cat, role in s.get("roles", {}).items()]
        )

        # Relationships
        rels_str = ", ".join(
            [
                f"{r['target']}({r['relationship_type']}-{r['description']})"
                for r in s.get("relationships", [])
            ]
        )

        # Painpoints
        pains_str = "; ".join(
            [f"{p['category']}-{p['description']}" for p in s.get("painpoints", [])]
        )

        summary = f'"{canonical}" | Roles: {roles_str}'
        if rels_str:
            summary += f" | Relates: {rels_str}"
        if pains_str:
            summary += f" | Pain: {pains_str}"

        summaries.append(f"{i}. {summary}")
    return summaries


# SINGLE BATCH PROCESSOR
async def process_hierarchy_batch(
    stakeholders: List[Dict], config: HierarchyConfig, batch_num: int
) -> List[Dict]:
    """Process one batch."""
    logger.info(f" Batch {batch_num}: {len(stakeholders)} stakeholders")

    summaries = build_stakeholder_summaries(stakeholders)
    prompt_text = f"""Batch {batch_num} - Analyze these stakeholders:

STAKEHOLDERS:
{chr(10).join(summaries)}

ECOSYSTEM LEVELS:
- macro: actors shaping system policy or governance (set policy/budget) → parent: null
- meso: actors implementing policy or coordinating operations (execute policy/regulate) → parent: macro   
- micro: operational actors, companies, groups, or individuals participating in the ecosystem (experience services) → parent: meso 

TASK: Assign EVERY stakeholder EXACTLY ONCE, NO DUPLICATES, NO EXTRAS:
1. primary_level: "macro" | "meso" | "micro" (dominant across all roles/relationships)
2. parent: The DIRECT supervising stakeholder from the list above.

RULES FOR parent:
- MUST be one of the stakeholder canonical_name values provided in the input list.
- MUST NOT invent new entities.
- If primary_level == "macro", parent MUST be null.

Output VALID JSON array only matching schema.
"""

    initial_state = InputState(topic=prompt_text, extraction_schema=HIERARCHY_SCHEMA)
    llm_config = Configuration(
        model=config.model,
        prompt="Assign hierarchy:\n{topic}\nSchema: {schema}\nJSON only.",
        max_loops=config.max_loops,
        temperature=0.0,
    ).__dict__

    try:
        final_state = await safe_llm_invoke(initial_state, llm_config)
        # print(f"\n\nfinal_state:{final_state}")
        assignments = parse_json_response(final_state.get("answer", "[]"))
        logger.info(
            f"  Batch {batch_num}: {len(assignments)}/{len(stakeholders)} assignments"
        )
    except Exception as e:
        logger.error(f"  Batch {batch_num} failed: {e}")
        assignments = []

    # mapping logic
    def normalize_key(name: str) -> str:
        return name.strip().lower()

    valid_names = {
        normalize_key(s["canonical_name"]): s["canonical_name"] for s in stakeholders
    }

    for h in assignments:
        parent = h.get("parent")
        if parent is not None:
            parent_norm = normalize_key(parent)
            if parent_norm == normalize_key(h["stakeholder"]):
                h["parent"] = None
            elif parent == "null":
                h["parent"] = None
            elif parent_norm not in valid_names:
                logger.warning(f"Invalid parent '{parent}'")
                h["parent"] = None

    assignment_map = {
        normalize_key(h["stakeholder"]): h for h in assignments
    }  # lowercase name-> stakholder info for assigned entities

    missing = [
        s["canonical_name"]
        for s in stakeholders  # input info
        if normalize_key(s["canonical_name"]) not in assignment_map
    ]
    if missing:
        logger.warning(f"  Batch {batch_num}, Entities not assigned by LLM: {missing}")

    result = []
    for stakeholder in stakeholders:
        canonical = stakeholder["canonical_name"]
        norm_key = normalize_key(canonical)
        assignment = assignment_map.get(
            norm_key, {"primary_level": "micro", "parent": None}
        )

        enriched = {
            "canonical_name": canonical,
            "roles": stakeholder.get("roles", {}),
            "relationships": stakeholder.get("relationships", []),
            "painpoints": stakeholder.get("painpoints", []),
            "primary_level": assignment["primary_level"],
            "parent": assignment.get("parent"),
        }
        result.append(enriched)

    return result


# MAIN BATCHING LOOP
async def assign_stakeholder_hierarchy(
    stakeholders: List[Dict], config: HierarchyConfig
) -> List[Dict]:
    """BATCHED version"""
    if len(stakeholders) <= config.batch_size:
        # Small input → single call (your original logic)
        logger.info(f"Single batch: {len(stakeholders)} stakeholders")
        return await process_hierarchy_batch(stakeholders, config, 1)

    # Large input → batching
    logger.info(
        f"Batching {len(stakeholders)} stakeholders (size: {config.batch_size})"
    )
    # print(f"\n\nstakeholders:{stakeholders}")
    all_results = []

    for batch_num in range(0, len(stakeholders), config.batch_size):
        batch = stakeholders[batch_num : batch_num + config.batch_size]
        batch_result = await process_hierarchy_batch(
            batch, config, batch_num // config.batch_size + 1
        )
        all_results.extend(batch_result)

    logger.info(f" Batching complete: {len(all_results)} stakeholders")
    return all_results


# ===== USAGE (unchanged) =====
async def main():
    parser = argparse.ArgumentParser(description="Stakeholder Hierarchy")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Input error: {e}")
        return

    # transform step
    transformed_data = transform_stakeholder_data(input_data)

    config = HierarchyConfig(
        output_dir=args.output_dir,
        batch_size=BATCH_SIZE,  # Pass batch size
    )
    hierarchy_output = await assign_stakeholder_hierarchy(transformed_data, config)

    input_filename = Path(args.input).name
    output_filename = input_filename.replace(".json", "_hierarchy.json")
    save_output(hierarchy_output, output_filename, args.output_dir)
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
