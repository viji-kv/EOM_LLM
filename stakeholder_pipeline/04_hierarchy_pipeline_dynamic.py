"""Stakeholder Hierarchy Pipeline - LLM Decides Hierarchy"""

import argparse
import asyncio
import json
import logging
import re
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

# ===== YOUR EXISTING IMPORTS =====
from dotenv import load_dotenv

from enrichment import graph
from enrichment.configuration import Configuration
from enrichment.state import InputState

load_dotenv()


@dataclass
class HierarchyConfig:
    model: str = "openai/gpt-4o-mini"
    max_loops: int = 2
    output_dir: str = "output"
    # Batch settings
    batch_size: int = BATCH_SIZE


# Dynamic Ecosystem Discovery Schema
ECOSYSTEM_DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "layers": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Layer name for the specific domain",
                    },
                    "description": {
                        "type": "string",
                        "description": "Precise role of this layer in hierarchy",
                        "maxLength": 100,
                    },
                },
                "required": ["name", "description"],
            },
        },
        "assignments": {
            "type": "array",
            "minItems": "{N}",  # Dynamic per batch
            "maxItems": "{N}",
            "uniqueItems": True,
            "items": {
                "type": "object",
                "properties": {
                    "stakeholder": {
                        "type": "string",
                        "description": "Exact canonical_name",
                    },
                    "layer": {
                        "type": "string",
                        "description": "Matches layers[i].name",
                    },
                    "parent": {
                        "type": ["string", "null"],
                        "description": "Supervising entity (null for macro roots). EXACT canonical_name from batch OR null. Must be higher level.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "1 sentence quoting summary evidence",
                        "maxLength": 150,
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Direct quote from stakeholder summary",
                        "maxLength": 60,
                    },
                },
                "required": ["stakeholder", "layer", "parent", "evidence"],
            },
        },
    },
    "required": ["layers", "assignments"],
}


def extract_json_from_response(text: str) -> Dict:
    """Tries multiple extraction strategies."""
    if not text:
        return {}

    # Standard ```json block (your case)
    match = re.search(r"```json?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        print("         Extracted from ```json block")
        json_str = match.group(1)
    # Any ``` block
    elif match := re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL):
        print("         Extracted from generic ``` block")
        json_str = match.group(1)
    # Largest JSON-like object {}
    elif match := re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        print("         Extracted largest JSON object")
        json_str = match.group(0)
    else:
        print(f"         No JSON found. Preview: {text[:200]}")
        return {}

    # Clean & parse
    json_str = (
        json_str.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").strip()
    )

    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            print(
                f"         SUCCESS: {{layers: {len(result.get('layers', []))}, assignments: {len(result.get('assignments', []))}}}"
            )
            return result
        else:
            print(f"         Not a dict: {type(result)}")
            return {}
    except json.JSONDecodeError as e:
        print(f"         Parse error: {e}")
        return {}


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


# Build dynamic schema with actual N
def build_dynamic_schema(stakeholders: List[Dict]) -> Dict:
    """Template {N} with actual batch size."""
    N = len(stakeholders)
    schema = ECOSYSTEM_DISCOVERY_SCHEMA.copy()
    if "minItems" in schema["properties"]["assignments"]:
        schema["properties"]["assignments"]["minItems"] = N
    if "maxItems" in schema["properties"]["assignments"]:
        schema["properties"]["assignments"]["maxItems"] = N
    return schema


# SINGLE BATCH PROCESSOR
async def process_hierarchy_batch(
    stakeholders: List[Dict], config: HierarchyConfig, batch_num: int
) -> List[Dict]:
    """Dynamic layers + richer output."""
    logger.info(f" Batch {batch_num}: {len(stakeholders)} stakeholders")

    summaries = build_stakeholder_summaries(stakeholders)
    dynamic_schema = build_dynamic_schema(stakeholders)

    prompt_text = f"""Discover ecosystem layers & assign stakeholders based on their Role and Relationship in the Ecosystem.:

STAKEHOLDERS:
{chr(10).join(summaries)}

MANDATORY: Assign ALL {len(stakeholders)} stakeholders exactly once.

TASK: 
1. DISCOVER 3-5 DOMAIN-SPECIFIC layers (top-down hierarchy).
2. Assign EVERY stakeholder EXACTLY ONCE to ONE layer.
3. Set parent: The DIRECT supervising stakeholder from the list above. (null for roots).

RULES:
- layers[].name must exactly match assignments[].layer
- parent: exact canonical_name from batch OR null
- Output VALID JSON matching schema ONLY.
- Base ALL reasoning STRICTLY on stakeholder summaries above. Quote evidence.
- NO external knowledge, invented collaborations, or future policies.

"""

    initial_state = InputState(topic=prompt_text, extraction_schema=dynamic_schema)
    llm_config = Configuration(
        model=config.model,
        prompt="Discover layers & assign:\n{topic}\nSchema: {schema}\nJSON only.",
        max_loops=config.max_loops,
        temperature=0.0,
    ).__dict__

    try:
        final_state = await safe_llm_invoke(initial_state, llm_config)
        # response = parse_json_response(final_state.get("answer", "{}"))
        response = extract_json_from_response(final_state.get("answer", "[]"))
        # print(f"final_state:{final_state}")
        layers = response.get("layers", [])
        assignments = response.get("assignments", [])
        logger.info(
            f"  Batch {batch_num}: {len(assignments)}/{len(stakeholders)} assigned, {len(layers)} layers"
        )
        logger.info(f"    Layers: {[l['name'] for l in layers]}")
    except Exception as e:
        logger.error(f"  Batch {batch_num} failed: {e}")
        layers, assignments = [], []

    # Layer-aware normalization & validation
    def normalize_key(name: str) -> str:
        return name.strip().lower()

    valid_names = {
        normalize_key(s["canonical_name"]): s["canonical_name"] for s in stakeholders
    }
    layer_map = {normalize_key(l["name"]): l for l in layers}
    # print(f"layer_map:{layer_map}")
    for h in assignments:
        parent = h.get("parent")
        if parent is not None:
            parent_norm = normalize_key(parent)
            if parent_norm == normalize_key(h["stakeholder"]):
                h["parent"] = None
            elif parent == "null":
                h["parent"] = None
            elif parent_norm not in valid_names:
                logger.warning(f"Invalid parent '{parent}' for {h['stakeholder']}")
                h["parent"] = None

    #  Safe access
    for h in assignments:
        layer = h.get("layer")  # Safe get
        if layer and normalize_key(layer) not in layer_map:
            logger.warning(f"Unknown layer '{layer}' for {h['stakeholder']}")

    assignment_map = {}
    for h in assignments:  # Now ONLY stakeholder objects
        if isinstance(h, dict) and h.get("stakeholder"):  # Stakeholder check
            assignment_map[normalize_key(h["stakeholder"])] = h
    # print(f"assignment_map:{assignment_map}")
    missing = [
        s["canonical_name"]
        for s in stakeholders
        if normalize_key(s["canonical_name"]) not in assignment_map
    ]
    if missing:
        logger.warning(f"  Batch {batch_num}, Entities not assigned by LLM: {missing}")

    # Enriched output with layers
    result = []
    for stakeholder in stakeholders:
        canonical = stakeholder["canonical_name"]
        norm_key = normalize_key(canonical)
        assignment = assignment_map.get(
            norm_key, {"layer": "unassigned", "parent": None, "reasoning": ""}
        )

        enriched = {
            "canonical_name": canonical,
            "roles": stakeholder.get("roles", {}),
            "relationships": stakeholder.get("relationships", []),
            "painpoints": stakeholder.get("painpoints", []),
            "layer_info": layer_map.get(normalize_key(assignment.get("layer", "")), {}),
            "parent": assignment.get("parent"),
            "reasoning": assignment.get("reasoning", ""),
            "evidence": assignment.get("evidence", ""),
        }
        result.append(enriched)

    return result


# MAIN BATCHING LOOP
async def assign_stakeholder_hierarchy(
    stakeholders: List[Dict], config: HierarchyConfig
) -> List[Dict]:
    """Batched dynamic layers."""
    if len(stakeholders) <= config.batch_size:
        logger.info(f"Single batch: {len(stakeholders)} stakeholders")
        return await process_hierarchy_batch(stakeholders, config, 1)

    logger.info(
        f"Batching {len(stakeholders)} stakeholders (size: {config.batch_size})"
    )
    all_results = []

    for batch_num in range(0, len(stakeholders), config.batch_size):
        batch = stakeholders[batch_num : batch_num + config.batch_size]
        batch_result = await process_hierarchy_batch(
            batch, config, batch_num // config.batch_size + 1
        )
        all_results.extend(batch_result)

    logger.info(f"Dynamic hierarchy complete: {len(all_results)} stakeholders")
    return all_results


# ===== USAGE  =====
async def main():
    parser = argparse.ArgumentParser(
        description="Stakeholder Hierarchy - Dynamic Layers"
    )
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
        batch_size=BATCH_SIZE,
    )
    hierarchy_output = await assign_stakeholder_hierarchy(transformed_data, config)

    input_filename = Path(args.input).name
    output_filename = input_filename.replace(".json", "_dynamic_hierarchy.json")
    save_output(hierarchy_output, output_filename, args.output_dir)
    logger.info("Dynamic Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
