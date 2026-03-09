"""
LLM-Only Stakeholder Hierarchy Classifier
Adapted from clustering script for macro/meso/micro + parents graph.
"""

import asyncio
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from collections import defaultdict
from stakeholder_pipeline.utils import parse_json_response, save_output

load_dotenv()

# ===== CONFIGURATION =====


@dataclass
class LLMHierarchyConfig:
    """Configuration for LLM-only hierarchy classification."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        output_dir: str = "output",
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_entities_per_prompt = 50  # Limit entities for hierarchy
        self.max_loops = 3  # More loops for complex graph reasoning


# ===== LLM HIERARCHY SCHEMA =====

HIERARCHY_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stakeholder canonical ID"},
                    "level": {
                        "type": "string",
                        "enum": ["macro", "meso", "micro"],
                        "description": "macro=gov/regulators, meso=operators/sectors, micro=firms",
                    },
                    "parent": {
                        "type": ["string", "null"],
                        "description": "Parent ID (upstream regulator/group) or null",
                    },
                    "size": {"type": "integer", "minimum": 10, "maximum": 50},
                },
                "required": ["id", "level", "parent", "size"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "weight": {"type": "integer", "default": 1},
                },
                "required": ["source", "target", "type"],
            },
        },
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False,
}

# ===== LLM HIERARCHY CLASSIFICATION =====


async def llm_classify_hierarchy(
    relationships: List[Dict], config: LLMHierarchyConfig
) -> Dict[str, Any]:
    """
    Use LLM to classify relationships into hierarchical graph.
    """
    from enrichment.state import InputState
    from enrichment.configuration import Configuration
    from enrichment import graph

    # Extract unique entities
    entities = list(set(e for r in relationships for e in [r["source"], r["target"]]))

    # Build prompt
    prompt_text = f"""Classify {len(entities)} stakeholders from these {len(relationships)} relationships into hierarchy:

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

UNIQUE ENTITIES (by index):"""

    for i, entity in enumerate(entities):
        prompt_text += f"\n{i}. {entity}"

    prompt_text += f"""

TASK: Build hierarchical graph.

CLASSIFICATION RULES:
- macro: governments/regulators (regulates/partners OUT, e.g. "LTA")
- meso: operators/sectors/groups (regulated BY macro, e.g. "Public Transport Operators") 
- micro: companies/projects (partners WITH meso, e.g. "SMRT")
- parent: upstream regulator/group (follows 'regulates' direction) or null
- A micro node must have a meso parent or null. A meso node must have a macro parent or null.
- size: 50(macro), 30(meso), 15(micro) - adjust by connections

Copy ALL edges exactly.

OUTPUT: Valid JSON matching schema ONLY."""

    initial_state = InputState(topic=prompt_text, extraction_schema=HIERARCHY_SCHEMA)
    llm_config = Configuration(
        model=config.model,
        prompt="Build stakeholder hierarchy graph:\n{topic}\nSchema: {schema}\nOutput valid JSON only.",
        max_loops=config.max_loops,
    ).__dict__

    print(f"       LLM classifying hierarchy for {len(entities)} entities...")

    final_state = await graph.ainvoke(initial_state, llm_config)
    result = parse_json_response(final_state.get("answer", ""))
    # print(result)
    # # Validate schema compliance
    # if not isinstance(result, dict) or "nodes" not in result or "edges" not in result:
    #     raise ValueError("Invalid hierarchy output - missing nodes/edges")

    return result


# ===== VALIDATE & ENRICH HIERARCHY =====


from collections import Counter


def weights_by_mentions(relationships: List[Dict]) -> List[Dict]:
    """
    Weight = # times exact (source,target,type) appears in source JSON.
    Handles duplicates naturally.
    """
    # Count mentions
    mention_counter = Counter(
        (r["source"], r["target"])
        for r in relationships  # , r["relationship_description"]
    )
    print(f"mention_counter:{mention_counter}")
    # Build weighted edges (deduplicated)
    weighted_edges = []
    for (src, tgt), count in mention_counter.items():
        weighted_edges.append(
            {
                "source": src,
                "target": tgt,
                # "type": typ,
                "weight": count,  # Direct mention count
                "evidence_strength": "high"
                if count >= 3
                else "medium"
                if count == 2
                else "low",
            }
        )
    print(f"weighted_edges:{weighted_edges}")
    return weighted_edges


# In your hierarchy script (validate_hierarchy):
def validate_hierarchy(
    hierarchy: Dict[str, Any], relationships: List[Dict]
) -> Dict[str, Any]:
    # print(f"hierarchy:{hierarchy}")
    if isinstance(hierarchy, list):
        print("INFO: LLM returned node list, inferring structure")
        hierarchy = {"nodes": hierarchy, "edges": []}
    hierarchy["edges"] = weights_by_mentions(relationships)
    print(f"        Weighted by mentions: {len(hierarchy['edges'])} unique edges")
    return hierarchy


# ===== MAIN PIPELINE =====


async def build_hierarchy_from_rels(
    relationships: List[Dict], config: LLMHierarchyConfig
) -> Dict[str, Any]:
    """
    LLM hierarchy pipeline from relationships.
    """
    if config is None:
        config = LLMHierarchyConfig()

    print(f" LLM-Only Hierarchy: {len(relationships)} relationships")

    # Single call (no categories/batching needed for rels)
    if len(relationships) > config.max_entities_per_prompt * 2:  # Rough token estimate
        print("     WARNING: Large input - consider splitting")

    hierarchy = await llm_classify_hierarchy(relationships, config)
    print(f"hierarchy:{hierarchy}")
    validated_hierarchy = validate_hierarchy(hierarchy, relationships)

    return {
        "hierarchy_graph": validated_hierarchy,
        "input_stats": {"relationships_count": len(relationships)},
        "processing_stats": {"llm_calls": 1},  # Single pass
    }


# ===== FILE I/O =====


async def hierarchy_from_rel_file(input_file: str, config: LLMHierarchyConfig) -> Dict:
    """Load rel JSON, build hierarchy, return result."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle direct list or wrapped
    relationships = data.get("relationships", data) if isinstance(data, dict) else data

    result = await build_hierarchy_from_rels(relationships, config)

    return {
        "brain": data.get("brain") if isinstance(data, dict) else None,
        "brain_id": data.get("brain_id") if isinstance(data, dict) else None,
        **result,
    }


def main():
    # parser = argparse.ArgumentParser(
    #     description="Build hierarchy from relationships JSON"
    # )
    # parser.add_argument("input_file", help="relationshipsoutput_<brain>.json")
    # parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model")
    # args = parser.parse_args()

    input_file = sys.argv[1]

    import time

    start = time.time()

    config = LLMHierarchyConfig(
        model="openai/gpt-4o-mini",
        output_dir="output",
    )

    # Run hierarchy classification
    result = asyncio.run(hierarchy_from_rel_file(input_file, config))

    elapsed = time.time() - start

    # Save output
    input_filename = Path(input_file).name
    output_filename = input_filename.replace(".json", "_hierarchy.json")
    output_path = save_output(result, output_filename, config.output_dir)

    # Print summary
    graph = result["hierarchy_graph"]
    print(f"graph:{graph.keys()}")
    print(f"\n LLM Hierarchy Complete")
    # print(f"    Nodes: {len(graph['nodes'])} ({dict(graph['stats']['levels'])})")
    # print(f"    Edges: {len(graph['edges'])}")
    # print(f"     Time: {elapsed:.1f}s")
    # print(f"    Saved: {output_path}")


if __name__ == "__main__":
    main()


# hierarchy = [
#     {
#         "id": "Monetary Authority of Singapore",
#         "level": "macro",
#         "parent": None,
#         "size": 50,
#     },
#     {
#         "id": "Infocomm Media Development Authority",
#         "level": "macro",
#         "parent": None,
#         "size": 50,
#     },
#     {
#         "id": "DBS Group Holdings",
#         "level": "meso",
#         "parent": "Monetary Authority of Singapore",
#         "size": 30,
#     },
# ]

# weighted_edges = [
#     {
#         "source": "HSBC",
#         "target": "Hong Kong Alzheimer's Disease Association",
#         "weight": 1,
#         "evidence_strength": "low",
#     },
#     {
#         "source": "DBS Group Holdings",
#         "target": "Singapore Centre for Social Enterprise",
#         "weight": 1,
#         "evidence_strength": "low",
#     },
#     {
#         "source": "DBS Group Holdings",
#         "target": "Community Chest",
#         "weight": 1,
#         "evidence_strength": "low",
#     },
# ]

# hierarchy.append(weighted_edges)
