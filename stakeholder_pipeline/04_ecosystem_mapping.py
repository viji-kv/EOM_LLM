"""
Stakeholder Hierarchy LLM Pipeline
Input: Your exact JSON above
Output: Same format + primary_level + parent
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from stakeholder_pipeline.utils import save_output
from tenacity import retry, stop_after_attempt, wait_exponential

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

    print("🤖 LLM analyzing ecosystem hierarchy...")
    final_state = await graph.ainvoke(initial_state, llm_config)

    hierarchy_assignments = parse_json_response(final_state.get("answer", ""))

    # Valid stakeholder names
    valid_names = {s["canonical_name"] for s in stakeholders}

    # Validate parents
    for h in hierarchy_assignments:
        parent = h.get("parent")

        if parent is not None and parent not in valid_names:
            print(f" Invalid parent detected: {parent}. Setting to None.")
            h["parent"] = None

        if parent == h.get("stakeholder"):
            h["parent"] = None

    # Map back to your stakeholders (preserves order)
    # assignment_map = {h["stakeholder"]: h for h in hierarchy_assignments}
    assignment_map = {
        h["stakeholder"].strip().lower(): h for h in hierarchy_assignments
    }

    # for stakeholder in stakeholders:
    #     canonical = stakeholder["canonical_name"].strip().lower()
    #     if canonical not in assignment_map:
    #         raise ValueError(f"LLM missing hierarchy assignment for: {canonical}")

    result = []
    for stakeholder in stakeholders:
        canonical = stakeholder["canonical_name"]
        look_up = stakeholder["canonical_name"].strip().lower()
        if look_up not in assignment_map:
            raise ValueError(f"LLM missing hierarchy assignment for: {canonical}")
        assignment = assignment_map.get(
            look_up, {"primary_level": "unknown", "parent": None}
        )

        # EXACT output format requested
        enriched = {
            "canonical_name": canonical,
            "roles": stakeholder.get("roles", {}),
            "relationships": stakeholder.get("relationships", []),
            "painpoints": stakeholder.get("painpoints", []),
            "primary_level": assignment["primary_level"],
            "parent": assignment["parent"],
        }

        result.append(enriched)

    print(f"✅ Hierarchy assigned to {len(result)} stakeholders")
    return result


# ===== USAGE =====
async def main():
    # Your consolidated stakeholders input
    input_file = "output/test_policy_output_transformed.json"
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    config = HierarchyConfig()
    hierarchy_output = await assign_stakeholder_hierarchy(input_data, config)

    # Save output
    input_filename = Path(input_file).name
    output_filename = input_filename.replace(".json", "_hierarchy.json")
    save_output(hierarchy_output, output_filename, "output")
    print(f"\n LLM Hierarchy Complete")


# RUN: python hierarchy.py
if __name__ == "__main__":
    asyncio.run(main())


# input_file = "output/test_policy_output_transformed.json"
# with open(input_file, "r") as f:
#     input_data = json.load(f)
