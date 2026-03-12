"""
Stakeholder Hierarchy LLM Pipeline

"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from stakeholder_pipeline.utils import save_output
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import argparse
from stakeholder_pipeline.transform_stakeholders import transform_stakeholder_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hierarchy.log")],
)
logger = logging.getLogger(__name__)

# ===== YOUR EXISTING IMPORTS (copy from your clustering code) =====
from stakeholder_pipeline.utils import parse_json_response
from enrichment.state import InputState
from enrichment.configuration import Configuration
from enrichment import graph
from dotenv import load_dotenv

load_dotenv()


@dataclass
class HierarchyConfig:
    model: str = "openai/gpt-4o-mini"
    max_loops: int = 2
    output_dir: str = "output"


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
    """Retry LLM calls."""
    try:
        return await graph.ainvoke(initial_state, llm_config)
    except Exception as e:
        logger.warning(f"LLM retry {e}")
        raise


async def assign_stakeholder_hierarchy(
    stakeholders: List[Dict], config: HierarchyConfig
) -> List[Dict]:
    """
    Single LLM call assigns hierarchy to your exact input format.
    """

    # Build summaries from YOUR data structure
    summaries = []
    for i, s in enumerate(stakeholders):
        canonical = s["canonical_name"]

        # Roles
        roles_str = "; ".join(
            [f"{cat}: {role}" for cat, role in s.get("roles", {}).items()]
        )

        # Relationships
        rels_str = ", ".join(
            [
                f"{r['target']}({r['relationship_type']})"
                for r in s.get("relationships", [])
            ]
        )

        # Painpoints
        pains_str = "; ".join(
            [
                f"{p['category']}: {p['description']}..."  # [:40]
                for p in s.get("painpoints", [])
            ]
        )

        summary = f'"{canonical}" | Roles: {roles_str}'
        if rels_str:
            summary += f" | Relates: {rels_str}"
        if pains_str:
            summary += f" | Pain: {pains_str}"

        summaries.append(f"{i}. {summary}")

    prompt_text = f"""Analyze this stakeholder ecosystem:

STAKEHOLDERS:
{chr(10).join(summaries)}

ECOSYSTEM LEVELS:
- macro: actors shaping system policy or governance (set policy/budget) → parent: null
- meso: actors implementing policy or coordinating operations (execute policy/regulate) → parent: macro   
- micro: operational actors, companies, groups, or individuals participating in the ecosystem (experience services) → parent: meso 

TASK: Assign EVERY stakeholder EXACTLY ONCE:
1. primary_level: "macro" | "meso" | "micro" (dominant across all roles/relationships)
2. parent: supervising entity (use EXACT canonical_name, null for roots)

Examples:
- Ministry of Transport → macro, parent: null
- Land Transport Authority → meso, parent: "Ministry of Transport"  
- SMRT → micro, parent: "Land Transport Authority"
- Commuters → micro, parent: "Land Transport Authority"

Output VALID JSON array only matching schema.
"""

    # YOUR existing LLM pipeline
    initial_state = InputState(topic=prompt_text, extraction_schema=HIERARCHY_SCHEMA)

    llm_config = Configuration(
        model=config.model,
        prompt="Assign stakeholder hierarchy:\n{topic}\nSchema: {schema}\nJSON only.",
        max_loops=config.max_loops,
    ).__dict__

    try:
        logger.info(" Calling LLM...")
        final_state = await safe_llm_invoke(initial_state, llm_config)
        hierarchy_assignments = parse_json_response(final_state.get("answer", "[]"))
        logger.info(f"LLM returned {len(hierarchy_assignments)} assignments")
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        hierarchy_assignments = []

    def normalize_key(name: str) -> str:
        return name.strip().lower()

    # Valid stakeholder names

    valid_names = {
        normalize_key(s["canonical_name"]): s["canonical_name"] for s in stakeholders
    }
    # print(f"valid_names:{valid_names}")
    # print(f"hierarchy_assignments:{hierarchy_assignments}")

    for h in hierarchy_assignments:
        stakeholder_norm = normalize_key(h["stakeholder"])
        parent = h.get("parent")

        if parent is not None:
            parent_norm = normalize_key(parent)
            if parent_norm == stakeholder_norm:
                logger.debug(f"Fixed self-parent: {h['stakeholder']}")
                h["parent"] = None
            elif parent_norm not in valid_names:
                logger.warning(f"Invalid parent '{parent}' → null")
                h["parent"] = None

    # 3. Strict mapping (fail if missing)
    assignment_map = {normalize_key(h["stakeholder"]): h for h in hierarchy_assignments}
    # print(f"assignment_map:{assignment_map}")

    # 4. STRICT: Fail-fast if LLM misses any
    missing = []
    for stakeholder in stakeholders:
        norm_key = normalize_key(stakeholder["canonical_name"])
        if norm_key not in assignment_map:
            missing.append(stakeholder["canonical_name"])

    if missing:
        logger.warning(
            f"  LLM MISSING {len(missing)}/{len(stakeholders)} assignments: {missing}"
        )
        logger.info("Continuing with partial LLM results...")
    else:
        logger.info(" 100% LLM coverage!")

    # 5. Clean enrichment
    result = []
    for stakeholder in stakeholders:
        canonical = stakeholder["canonical_name"]
        norm_key = normalize_key(canonical)
        # assignment = assignment_map[norm_key]
        assignment = assignment_map.get(norm_key)
        if not assignment:
            logger.warning(f"No assignment for {canonical}")
            assignment = {"primary_level": "micro", "parent": None}

        enriched = {
            "canonical_name": canonical,
            "roles": stakeholder.get("roles", {}),
            "relationships": stakeholder.get("relationships", []),
            "painpoints": stakeholder.get("painpoints", []),
            "primary_level": assignment["primary_level"],
            "parent": assignment.get("parent"),
        }
        result.append(enriched)

    # logger.info(f" STRICT mapping: {len(result)} stakeholders (100% LLM coverage)")
    llm_success = (len(stakeholders) - len(missing)) / len(stakeholders) * 100
    logger.info(
        f" Pipeline complete: {len(result)} stakeholders, {llm_success:.1f}% LLM success"
    )
    return result


# ===== USAGE =====
async def main():
    parser = argparse.ArgumentParser(description="Stakeholder Hierarchy")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")

    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    # input_file = "output/test_policy_output_transformed.json"
    # with open(input_file, "r", encoding="utf-8") as f:
    #     input_data = json.load(f)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Input error: {e}")
        return

    transformed_data = transform_stakeholder_data(input_data)

    config = HierarchyConfig(output_dir=args.output_dir)
    hierarchy_output = await assign_stakeholder_hierarchy(transformed_data, config)

    # Save output
    # input_filename = Path(input_file).name
    input_filename = Path(args.input).name
    output_filename = input_filename.replace(".json", "_hierarchy.json")
    save_output(hierarchy_output, output_filename, args.output_dir)
    logger.info(" Pipeline complete!")


# RUN: python hierarchy.py
if __name__ == "__main__":
    asyncio.run(main())
