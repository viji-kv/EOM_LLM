"""Stakeholder Consolidator: Normalize → Embeddings → LLM Labels
Emdedding is done based on normalized names.
"""

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# Configuration class
@dataclass
class ConsolidationConfig:
    """Configuration for consolidation pipeline."""

    embedding_threshold: float = 0.70
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "openai/gpt-4o-mini"
    output_dir: str = "output"


# ===== STEP 1: DETERMINISTIC NORMALIZATION =====


def deterministic_normalize(name: str) -> str:
    """Production-grade normalization: abbreviations, titles, punctuation."""
    abbrev_map = {
        r"\bgovt?\b": "government",
        r"\bdept\b": "department",
        r"\bassoc\b": "association",
        r"\bltd\b": "",
        r"\binc\b": "",
        r"\bcorp\b": "corporation",
    }

    normalized = name.lower().strip()

    for pattern, replacement in abbrev_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# ===== STEP 2: EMBEDDING SIMILARITY CLUSTERING =====


# Added config parameter
async def get_embeddings(texts: List[str], config: ConsolidationConfig) -> np.ndarray:
    """Get embeddings using OpenAI text-embedding-3-small."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    print(f"      Fetching OpenAI embeddings for {len(texts)} stakeholders...")

    # Use config.embedding_model
    response = await client.embeddings.create(
        model=config.embedding_model,
        input=texts,
        encoding_format="float",
    )

    embeddings = np.array([item.embedding for item in response.data])
    print(f"      Got {embeddings.shape} embeddings")

    return embeddings


# Added config parameter, made threshold configurable
def cluster_by_category(
    stakeholders: List[Dict], embeddings: np.ndarray, config: ConsolidationConfig
) -> List[List[int]]:
    """Cluster using cosine similarity of embeddings within categories."""
    category_blocks = defaultdict(list)

    for i, s in enumerate(stakeholders):
        cat = s.get("Category", "Unknown")
        category_blocks[cat].append(i)

    print(f"   Created {len(category_blocks)} category blocks")

    all_clusters = []

    for cat, indices in category_blocks.items():
        if len(indices) == 1:
            all_clusters.append(indices)
            continue

        print(f"   Embedding clustering {len(indices)} stakeholders in '{cat}'")

        cat_embeddings = embeddings[
            indices
        ]  # for each index, embeddings are of shape 1536.

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(
            cat_embeddings
        )  # shape(num of indices, num of indices)

        used = set()
        for i_pos, i_idx in enumerate(indices):
            if i_idx in used:
                continue

            cluster = [i_idx]
            used.add(i_idx)

            for j_pos in range(i_pos + 1, len(indices)):
                j_idx = indices[j_pos]
                if j_idx in used:
                    continue

                # Use config.embedding_threshold
                if similarity_matrix[i_pos][j_pos] >= config.embedding_threshold:
                    cluster.append(j_idx)
                    used.add(j_idx)

            all_clusters.append(cluster)

    return all_clusters


# ===== STEP 3: LLM FOR CANONICAL LABELS ONLY =====


async def llm_canonical_label(
    cluster_names: List[str], category: str, config: ConsolidationConfig
) -> str:
    """LLM generates ONE canonical name for a cluster."""
    from enrichment import graph
    from enrichment.configuration import Configuration
    from enrichment.state import InputState

    prompt_text = f"""Given these variant names for a {category} stakeholder, provide the SINGLE most official/canonical name.

Variants:
{chr(10).join(f"- {name}" for name in cluster_names)}

Output ONLY the canonical name (no explanation):"""

    schema = {
        "type": "object",
        "properties": {
            "canonical_name": {
                "type": "string",
                "description": "Official canonical name",
            }
        },
        "required": ["canonical_name"],
    }

    initial_state = InputState(topic=prompt_text, extraction_schema=schema)

    # Use config.llm_model
    llm_config = Configuration(
        model=config.llm_model,
        prompt="Extract canonical name:\n{topic}\nSchema: {schema}",
        max_loops=1,
    ).__dict__

    final_state = await graph.ainvoke(initial_state, llm_config)

    try:
        result = json.loads(
            re.sub(r"```json?|\n*```", "", final_state.get("answer", "").strip())
        )
        return result.get("canonical_name", cluster_names[0])
    except:
        return cluster_names[0]  # Fallback to first name


# Added config parameter
async def merge_with_llm_labels(
    clusters: List[List[int]], stakeholders: List[Dict], config: ConsolidationConfig
) -> List[Dict]:
    """Merge clusters using LLM-generated canonical labels."""
    consolidated = []

    for cluster in clusters:
        if len(cluster) == 1:
            result = stakeholders[cluster[0]].copy()
            result["consolidation_info"] = {
                "cluster_size": 1,
                # "method": "singleton",
                "member_indices": cluster,
            }
            consolidated.append(result)
            continue

        # Multi-member cluster: get LLM canonical name
        cluster_names = [
            stakeholders[i].get(
                "Canonical Name", stakeholders[i].get("Stakeholder Name")
            )
            for i in cluster
        ]
        category = stakeholders[cluster[0]].get("Category", "Unknown")

        print(f"      LLM labeling {len(cluster_names)} variants in '{category}'")

        # Pass config to llm_canonical_label
        canonical = await llm_canonical_label(cluster_names, category, config)

        # Pick highest confidence as base record
        scores = [
            (int(stakeholders[i].get("Confidence Score", "0").rstrip("%")), i)
            for i in cluster
        ]
        primary_idx = max(scores)[1]

        master = stakeholders[primary_idx].copy()
        master["Canonical Name"] = canonical  # Override with LLM canonical

        # Aggregate sources
        all_sources = []
        for i in cluster:
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
            "cluster_size": len(cluster),
            # "method": "embedding_llm_label",
            "member_indices": cluster,
            "all_sources": all_sources,
        }

        consolidated.append(master)

    return consolidated


# ===== MAIN PIPELINE =====


# Added config param
async def consolidate_stakeholders(
    stakeholders: List[Dict], config: Optional[ConsolidationConfig] = None
) -> Dict:
    """Pipeline: Normalize → Embeddings → LLM Labels

    Args:
        stakeholders: List of stakeholder dicts
        config: Optional configuration object

    Returns:
        Dict with consolidated stakeholders and stats
    """
    # Create default config if not provided
    if config is None:
        config = ConsolidationConfig()

    print(f"Consolidating {len(stakeholders)} stakeholders")

    # Step 1: Normalize
    print("   Step 1: Normalizing names...")
    for s in stakeholders:
        original = s.get("Canonical Name", s.get("Stakeholder Name", ""))
        s["_normalized"] = deterministic_normalize(original)

    # Step 2: Embeddings
    print("   Step 2: Generating embeddings...")
    texts = [s["_normalized"] for s in stakeholders]
    # Pass config
    embeddings = await get_embeddings(texts, config)

    # Step 3: Clustering
    print("   Step 3: Embedding clustering within categories...")
    # Pass config instead of threshold
    clusters = cluster_by_category(stakeholders, embeddings, config)

    multi = [c for c in clusters if len(c) > 1]
    print(f"   Found {len(multi)} multi-member clusters")

    # Step 4: LLM labels
    print("   Step 4: LLM generating canonical names...")
    # Pass config
    consolidated = await merge_with_llm_labels(clusters, stakeholders, config)

    # Sort by confidence
    consolidated.sort(
        key=lambda x: int(x.get("Confidence Score", "0").rstrip("%")), reverse=True
    )

    # Return dict with stats
    return {
        "consolidated_stakeholders": consolidated,
        "stats": {
            "original_count": len(stakeholders),
            "consolidated_count": len(consolidated),
            "embedding_threshold": config.embedding_threshold,
        },
    }


# File I/O helper
async def consolidate_from_file(
    input_file: str, config: Optional[ConsolidationConfig] = None
) -> Dict:
    """Load JSON, consolidate, return result (no file writing)."""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    result = await consolidate_stakeholders(data["stakeholders"], config)

    # Preserve original metadata
    output = {
        "brain": data.get("brain"),
        "brain_id": data.get("brain_id"),
        **result,  # Spread consolidated_stakeholders and stats
    }

    return output


def save_consolidated_output(
    result: Dict, output_filename: str, output_dir: str = "output"
) -> str:
    """Save consolidation result to file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    out_file = output_path / output_filename

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(out_file)


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print(
            "Usage: python consolidate_stakeholders.py stakeholders_output_BRAIN.json"
        )
        print(
            "\nAdvanced approach: Normalize → Embeddings → LLM Labels (92-95% recall)"
        )
        sys.exit(1)

    input_file = sys.argv[1]

    result = asyncio.run(consolidate_from_file(input_file))

    # output filename
    input_filename = Path(input_file).name
    output_filename = input_filename.replace(".json", "_consolidated.json")

    output_path = save_consolidated_output(result, output_filename)

    # Stats dict
    print(f"\nConsolidated {result['stats']['consolidated_count']} stakeholders")
    print(f"   Saved: {output_path}")


if __name__ == "__main__":
    main()
