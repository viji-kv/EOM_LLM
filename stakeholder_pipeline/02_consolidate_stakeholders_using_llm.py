"""
LLM-Only Stakeholder Clustering

"""

import asyncio
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from collections import defaultdict
from stakeholder_pipeline.utils import parse_json_response, save_output


load_dotenv()

# ===== CONFIGURATION =====


@dataclass
class LLMClusterConfig:
    """Configuration for LLM-only clustering."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        output_dir: str = "output",
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_stakeholders_per_prompt = 100  # Limit to 100 stakeholders
        self.max_loops = 2  # Limit to 2 loops for testing (initial + 1 retry if needed)


# ===== LLM CLUSTERING SCHEMA =====

CLUSTERING_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "cluster_id": {"type": "integer", "description": "Unique cluster ID"},
            "canonical_name": {
                "type": "string",
                "description": "Official name for this cluster",
            },
            "member_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of stakeholder indices that belong together",
            },
        },
        "required": ["cluster_id", "canonical_name", "member_indices"],
    },
}


# ===== LLM CLUSTERING =====


async def llm_cluster_stakeholders(
    stakeholders: List[Dict], category: str, config: LLMClusterConfig
) -> List[Dict]:
    """
    Use LLM to cluster stakeholders within a category.
    """
    from enrichment.state import InputState
    from enrichment.configuration import Configuration
    from enrichment import graph

    # Build prompt with all stakeholders
    prompt_text = f"""You are clustering {len(stakeholders)} {category} stakeholders to identify duplicates.

STAKEHOLDERS (by index):
"""

    for i, s in enumerate(stakeholders):
        name = s.get("Canonical Name", s.get("Stakeholder Name", "Unknown"))
        role = s.get("Role", "")[:100]  # Truncate long roles
        conf = s.get("Confidence Score", "N/A")
        # prompt_text += f"{i}. {name} - {role} (conf: {conf})\n"
        prompt_text += f"{i}. {name}\n"

    prompt_text += f"""

TASK: Group stakeholders that represent the SAME entity (duplicates/variants).

RULES:
Same organization with different names → CLUSTER (e.g., "SWD" + "Social Welfare Dept")
Each stakeholder must appear in EXACTLY ONE cluster
Singletons (no duplicates) get their own cluster
Please provide your answer directly in clear text, filling in the schema.

OUTPUT: JSON array of clusters with canonical names.
"""

    initial_state = InputState(topic=prompt_text, extraction_schema=CLUSTERING_SCHEMA)
    # print(f"prompt_text: {prompt_text}")
    llm_config = Configuration(
        model=config.model,
        prompt="Cluster stakeholders:\n{topic}\nSchema: {schema}\nOutput valid JSON only.",
        max_loops=config.max_loops,
    ).__dict__

    print(f"       LLM clustering {len(stakeholders)} {category} stakeholders...")

    final_state = await graph.ainvoke(initial_state, llm_config)
    # print(f"Final state: {final_state}")

    result = parse_json_response(final_state.get("answer", ""))
    # print(f"Parsed clusters: {result}")
    return result


# ===== MERGE CLUSTERS =====


def merge_llm_clusters(
    clusters: List[Dict],
    stakeholders: List[Dict],
    original_indices: List[int],  # Pass actual indices for this category
) -> List[Dict]:
    """
    Merge stakeholders based on LLM clustering with STRICT validation.
    """
    consolidated = []
    used_indices = set()

    # Create mapping from LLM indices (0,1,2...) to real indices
    index_map = {i: real_idx for i, real_idx in enumerate(original_indices)}
    max_llm_index = len(original_indices) - 1
    print(f"      Validating LLM clusters (max index: {max_llm_index})...")

    for cluster_info in clusters:
        member_indices = cluster_info.get("member_indices", [])

        if not member_indices:
            continue

        # Validate ALL indices before processing
        valid_llm_indices = []
        invalid_count = 0

        for llm_idx in member_indices:
            if not isinstance(llm_idx, int):
                invalid_count += 1
                continue
            if llm_idx < 0 or llm_idx > max_llm_index:
                print(
                    f"        LLM hallucinated index {llm_idx} (max is {max_llm_index})"
                )
                invalid_count += 1
                continue
            if llm_idx in valid_llm_indices:
                print(f"        Duplicate index {llm_idx} in cluster")
                continue
            valid_llm_indices.append(llm_idx)
        if invalid_count > 0:
            print(
                f"        Cluster {cluster_info.get('cluster_id')} had {invalid_count} invalid indices"
            )

        if not valid_llm_indices:
            continue

        # Map to real stakeholder indices
        real_indices = [index_map[i] for i in valid_llm_indices if i in index_map]

        # Check for already used indices (LLM might duplicate)
        real_indices = [i for i in real_indices if i not in used_indices]

        if not real_indices:
            print(
                f"        Cluster {cluster_info.get('cluster_id')} has no valid/unused indices"
            )
            continue

        used_indices.update(real_indices)

        # Pick highest confidence as primary
        scores = [
            (int(stakeholders[i].get("Confidence Score", "0").rstrip("%")), i)
            for i in real_indices
        ]
        primary_idx = max(scores)[1]

        master = stakeholders[primary_idx].copy()
        master["Canonical Name"] = cluster_info.get(
            "canonical_name", master.get("Canonical Name")
        )

        # Aggregate sources
        all_sources = []
        for i in real_indices:
            meta = stakeholders[i].get("Source metadata", {})
            all_sources.append(
                {
                    "index": i,
                    "original_name": stakeholders[i].get("Stakeholder Name"),
                    "filename": meta.get("filename"),
                    "confidence": stakeholders[i].get("Confidence Score"),
                }
            )

        master["consolidation_info"] = {
            "cluster_size": len(real_indices),
            "member_indices": real_indices,
            "all_sources": all_sources,
        }

        consolidated.append(master)

    # Add ONLY unused stakeholders from original_indices
    for real_idx in original_indices:
        if real_idx not in used_indices:
            singleton = stakeholders[real_idx].copy()
            singleton["consolidation_info"] = {
                "cluster_size": 1,
                "member_indices": [real_idx],
            }
            consolidated.append(singleton)
            used_indices.add(real_idx)

    print(
        f"       Created {len(consolidated)} consolidated (from {len(original_indices)} original)"
    )

    return consolidated


# ===== MAIN PIPELINE =====


async def consolidate_with_llm_only(
    stakeholders: List[Dict], config: LLMClusterConfig
) -> Dict:
    """
    Pure LLM clustering pipeline (for comparison testing).
    """
    if config is None:
        config = LLMClusterConfig()

    print(f" LLM-Only Clustering: {len(stakeholders)} stakeholders")

    # Group by category (still block by category for tractability)
    category_blocks = defaultdict(list)
    for i, s in enumerate(stakeholders):
        cat = s.get("Category", "Unknown")
        category_blocks[cat].append(i)

    print(f"    Processing {len(category_blocks)} categories")

    all_consolidated = []

    for cat, indices in category_blocks.items():
        print(f"\n   Category: {cat} ({len(indices)} stakeholders)")

        # Extract stakeholders for this category
        cat_stakeholders = [stakeholders[i] for i in indices]

        if len(cat_stakeholders) == 1:
            # Singleton category
            single = cat_stakeholders[0].copy()
            single["consolidation_info"] = {
                "cluster_size": 1,
                "member_indices": indices,
            }
            all_consolidated.append(single)
            continue

        # Split if too many (avoid context overflow)
        if len(cat_stakeholders) > config.max_stakeholders_per_prompt:
            print(
                f"        Category too large ({len(cat_stakeholders)}), splitting into batches..."
            )

            batch_size = config.max_stakeholders_per_prompt

            for batch_idx in range(0, len(cat_stakeholders), batch_size):
                # Extract batch of stakeholders
                batch = cat_stakeholders[batch_idx : batch_idx + batch_size]

                # Extract corresponding indices (same slice)
                batch_indices = indices[batch_idx : batch_idx + batch_size]

                batch_num = batch_idx // batch_size + 1
                total_batches = (len(cat_stakeholders) - 1) // batch_size + 1

                print(
                    f"      Batch {batch_num}/{total_batches}: {len(batch)} stakeholders (indices {batch_indices[0]}-{batch_indices[-1]})"
                )

                # Cluster this batch
                clusters = await llm_cluster_stakeholders(batch, cat, config)

                # Merge with validation (pass full stakeholder list + batch indices)
                merged = merge_llm_clusters(clusters, stakeholders, batch_indices)

                all_consolidated.extend(merged)
        else:
            # Process whole category (no batching needed)
            clusters = await llm_cluster_stakeholders(cat_stakeholders, cat, config)

            # Pass full indices list for this category
            merged = merge_llm_clusters(clusters, stakeholders, indices)

            all_consolidated.extend(merged)

    # Sort by confidence
    all_consolidated.sort(
        key=lambda x: int(x.get("Confidence Score", "0").rstrip("%")), reverse=True
    )

    return {
        "consolidated_stakeholders": all_consolidated,
        "stats": {
            "original_count": len(stakeholders),
            "consolidated_count": len(all_consolidated),
            "categories_processed": len(category_blocks),
        },
    }


# ===== FILE I/O =====


async def consolidate_from_file(input_file: str, config: LLMClusterConfig) -> Dict:
    """Load JSON, cluster with LLM, return result."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = await consolidate_with_llm_only(data["stakeholders"], config)

    return {
        "brain": data.get("brain"),
        "brain_id": data.get("brain_id"),
        "consolidation_stats": result["stats"],
        "consolidated_stakeholders": result["consolidated_stakeholders"],
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python llm_only_clustering.py stakeholders_output.json")
        sys.exit(1)

    input_file = sys.argv[1]

    import time

    start = time.time()

    config = LLMClusterConfig(
        model="openai/gpt-4o-mini",
        output_dir="output",
    )

    # Run LLM-only clustering
    result = asyncio.run(consolidate_from_file(input_file, config))

    elapsed = time.time() - start

    # Save output
    input_filename = Path(input_file).name
    output_filename = input_filename.replace(".json", "_consolidated.json")
    output_path = save_output(result, output_filename, config.output_dir)

    # Print summary
    stats = result["consolidation_stats"]
    print(f"\n LLM-Only Clustering Complete")
    print(
        f"    Original: {stats['original_count']} → Consolidated: {stats['consolidated_count']}"
    )
    # print(f"    Reduction: {stats['reduction_pct']}%")
    print(f"     Time: {elapsed:.1f}s")
    print(f"    Saved: {output_path}")


if __name__ == "__main__":
    main()
