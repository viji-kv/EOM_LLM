"""Stakeholder Two-Level Clustering Pipeline"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from enrichment import graph
from enrichment.configuration import Configuration
from enrichment.state import InputState
from stakeholder_pipeline.utils import parse_json_response, save_output

load_dotenv()

#  WINDOWS UTF-8 FIX
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("clustering.log", encoding="utf-8"),  #  Critical
    ],
)
logger = logging.getLogger(__name__)


def flatten_ecosystem_to_stakeholder_records(
    ecosystem_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flatten ecosystem analysis JSON to per-stakeholder records with related pain points,
    relationships, themes, and theme clusters.
    """
    eco_root = ecosystem_data.get("current_analysis_result", {}).get(
        "ecosystem_analysis", {}
    )
    stakeholders = eco_root.get("stakeholders", [])
    pain_points = eco_root.get("pain_points", [])
    relationships = eco_root.get("relationships", [])
    themes = eco_root.get("themes", [])
    theme_clusters = eco_root.get("theme_clusters", [])

    # Create stakeholder name to record mapping for quick lookup
    stakeholder_records = {}

    for sh in stakeholders:
        sh_id = sh["id"]
        sh_name = sh["name"]
        stakeholder_records[sh_name] = {
            "stakeholder_id": sh_id,
            "stakeholder_name": sh_name,
            "stakeholder_long_name": sh["long_name"],
            "category": sh["category"],
            "role": sh["role"],
            "hierarchy_level": sh["hierarchy_level"],
            # "confidence": sh["confidence"],
            # "original_names": sh["original_names"],
            "influence_scope": sh.get("influence_scope"),
            "intervention_capacity": sh.get("intervention_capacity"),
            "decision_authority": sh.get("decision_authority"),
            "resource_control": sh.get("resource_control"),
            "challenge_relevance_capacity": sh.get("challenge_relevance_capacity"),
            "cross_theme_connections": sh.get("cross_theme_connections"),
            "mentions": sh["mentions"],
            # Initialize related data lists
            "related_pain_points": [],
            "related_relationships_source": [],
            "related_relationships_target": [],
            "related_themes": [],
            "related_theme_clusters": [],
        }

    # Attach pain point info
    for pp in pain_points:
        pp_base = {
            "pain_point_id": pp["id"],
            "label": pp["label"],
            "category_pain_point": pp["category"],
            "description_pain_point": pp["description"],
            "hierarchy_level_pain_point": pp["hierarchy_level"],
            "severity": pp["severity"],
            "urgency": pp["urgency"],
            "intervention_difficulty": pp["intervention_difficulty"],
            "causing_stakeholders": pp.get("causing_stakeholders", []),
        }

        impacts = pp.get("stakeholder_impact", {})

        # Affected link
        for name in pp.get("affected_stakeholders", []):
            if name in stakeholder_records:
                rec = stakeholder_records[name]
                row = {
                    **pp_base,
                    "pain_point_impact_text": impacts.get(name, ""),
                }
                rec["related_pain_points"].append(row)

    # Link relationships
    for rel in relationships:
        source = rel["source_stakeholder"]
        target = rel["target_stakeholder"]

        if source in stakeholder_records:
            stakeholder_records[source]["related_relationships_source"].append(
                {
                    "relationship_id": rel["id"],
                    "target": target,
                    "relationship_type": rel["relationship_type"],
                    "relationship_subtype": rel["relationship_subtype"],
                    "strength": rel["strength"],
                    "relationship_formality": rel["relationship_formality"],
                    "collaboration_potential": rel["collaboration_potential"],
                    "conflict_potential": rel["conflict_potential"],
                }
            )

        if target in stakeholder_records:
            stakeholder_records[target]["related_relationships_target"].append(
                {
                    "relationship_id": rel["id"],
                    "source": source,
                    "relationship_type": rel["relationship_type"],
                    "relationship_subtype": rel["relationship_subtype"],
                    "strength": rel["strength"],
                    "relationship_formality": rel["relationship_formality"],
                    "collaboration_potential": rel["collaboration_potential"],
                    "conflict_potential": rel["conflict_potential"],
                }
            )

    # Link themes (match by stakeholder name in related_stakeholders)
    for theme in themes:
        for sh_name in theme.get("related_stakeholders", []):
            if sh_name in stakeholder_records:
                stakeholder_records[sh_name]["related_themes"].append(
                    {
                        "theme_id": theme["id"],
                        "theme_name": theme["theme_name"],
                        "description": theme["description"],
                        "prevalence": theme["prevalence"],
                        "related_pain_points": theme["related_pain_points"],
                        "potential_challenges": theme["potential_challenges"],
                    }
                )

    # Link theme clusters (match themes within clusters)
    for cluster in theme_clusters:
        cluster_themes = cluster.get("themes", [])
        for sh_name, record in stakeholder_records.items():
            matching_themes = [
                t
                for t in cluster_themes
                if any(theme["theme_name"] == t for theme in record["related_themes"])
            ]
            if matching_themes:
                record["related_theme_clusters"].append(
                    {
                        "cluster_name": cluster["cluster_name"],
                        "cluster_description": cluster["cluster_description"],
                        "matching_themes": matching_themes,  # Keep only matching themes for each sh
                    }
                )

    # Convert to list and sort by stakeholder ID
    result = list(stakeholder_records.values())
    result.sort(key=lambda x: x["stakeholder_id"])

    return result


@dataclass
class ClusteringConfig:
    model: str = "openai/gpt-4o"
    max_loops: int = 1
    output_dir: str = "output"
    # Adaptive ranges
    level1_min: int = 4
    level1_max: int = 20
    level2_min: int = 2
    level2_max: int = 10


# LEVEL 1 SCHEMA
LEVEL1_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "n_clusters": {
            "type": "integer",
            "description": "Number of level1 clusters DECIDED by LLM",
            # "minimum": 4,
            # "maximum": 15,
        },
        "assignments": {
            "type": "array",
            "description": "EXACTLY n_clusters assignments, EVERY stakeholder ONCE",
            "items": {
                "type": "object",
                "properties": {
                    "stakeholder_id": {
                        "type": "string",
                        "description": "EXACT stakeholder_id from input (case-sensitive match)",
                    },
                    "stakeholder": {
                        "type": "string",
                        "description": "EXACT stakeholder_name from input (case-sensitive match)",
                    },
                    "cluster_id": {
                        "type": "integer",
                        "description": "0 to (n_clusters-1), NO duplicates across stakeholders",
                    },
                    "cluster_label": {
                        "type": "string",
                        "description": "Short descriptive name (1-3 words): 'Regulators', 'Vendors', 'End-Users'",
                        "maxLength": 30,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this stakeholder belongs here (role/pains/themes match)",
                        "maxLength": 150,
                    },
                },
                "required": [
                    "stakeholder_id",
                    "stakeholder",
                    "cluster_id",
                    "cluster_label",
                ],
            },
        },
    },
    "required": ["n_clusters", "assignments"],
    "additionalProperties": False,
}

# LEVEL 2 SCHEMA
LEVEL2_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "n_super_clusters": {
            "type": "integer",
            "description": "Number of level2 super-clusters DECIDED by LLM",
        },
        "super_assignments": {
            "type": "array",
            "description": "EXACTLY n_super_clusters assignments, EVERY level1 group ONCE",
            "items": {
                "type": "object",
                "properties": {
                    "level1_cluster": {
                        "type": "string",
                        "description": "EXACT cluster_label from level1 output",
                    },
                    "super_cluster_id": {
                        "type": "integer",
                        "description": "0 to (n_super_clusters-1)",
                    },
                    "super_cluster_label": {
                        "type": "string",
                        "description": "Descriptive super-group (2-4 words): 'Governance Actors', 'Operations Chain'",
                        "maxLength": 40,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why these level1 groups belong together (shared scope/influence)",
                        "maxLength": 150,
                    },
                },
                "required": [
                    "level1_cluster",
                    "super_cluster_id",
                    "super_cluster_label",
                ],
            },
        },
    },
    "required": ["n_super_clusters", "super_assignments"],
    "additionalProperties": False,
}


def build_level1_schema(config: ClusteringConfig) -> Dict[str, Any]:
    """Dynamic schema matching config ranges."""
    schema = LEVEL1_RESPONSE_SCHEMA.copy()
    schema["properties"]["n_clusters"]["minimum"] = config.level1_min
    schema["properties"]["n_clusters"]["maximum"] = config.level1_max
    return schema


def build_level2_schema(config: ClusteringConfig) -> Dict[str, Any]:
    """Dynamic schema matching config ranges."""
    schema = LEVEL2_RESPONSE_SCHEMA.copy()
    schema["properties"]["n_super_clusters"]["minimum"] = config.level2_min
    schema["properties"]["n_super_clusters"]["maximum"] = config.level2_max
    return schema


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def safe_llm_invoke(initial_state: InputState, llm_config: Dict) -> Dict:
    """Retry logic."""
    try:
        return await graph.ainvoke(initial_state, llm_config)
    except Exception as e:
        logger.warning(f"LLM invoke failed: {e}")
        raise


def build_stakeholder_summaries(stakeholders: List[Dict[str, Any]]) -> List[str]:
    """Consolidated, informative summaries."""
    summaries = []

    for i, s in enumerate(stakeholders):
        sid = s.get("stakeholder_id", "")
        name = s.get("stakeholder_name", "")
        core = (
            f"#{i} Stakeholder ID:{sid} | "
            f'Stakeholder Name:"{name}" | '
            # f'Long Name:"{s.get("stakeholder_long_name", "")}" | '
            f"Category:{s.get('category', '')} | "
            f"Role:{s.get('role', '')} | "
            f"Hierarchy Level:{s.get('hierarchy_level', '')} | "
            f"Influence Scope:{s.get('influence_scope', '')} | "
            f"Intervention Capacity:{s.get('intervention_capacity', '')} | "
            f"Decision Authority:{s.get('decision_authority', '')} | "
            f"Resource Control:{s.get('resource_control', '')} | "
            f"Challenge Relevance:{s.get('challenge_relevance_capacity', '')} | "
            f"Cross-theme Connections:{s.get('cross_theme_connections', '')} | "
            f"Mentions:{s.get('mentions', 0)}"
        )

        # Pain points: keep all, with IDs, labels, and key fields
        pain_chunks = []
        for pp in s.get("related_pain_points", []):
            pain_chunks.append(
                " | ".join(
                    [
                        f"PainPointID:{pp.get('pain_point_id', '')}",
                        f"Label:{pp.get('label', '')}",
                        f"Category:{pp.get('category_pain_point', '')}",
                        f"Description:{pp.get('description_pain_point', '')}",
                        f"Hierarchy:{pp.get('hierarchy_level_pain_point', '')}",
                        f"Severity:{pp.get('severity', '')}",
                        f"Urgency:{pp.get('urgency', '')}",
                        f"InterventionDifficulty:{pp.get('intervention_difficulty', '')}",
                        f"CausingStakeholders:{', '.join(pp.get('causing_stakeholders', []) or [])}",
                        f"Impact:{pp.get('pain_point_impact_text', '')}",
                    ]
                )
            )
        pains_block = " || ".join(pain_chunks) if pain_chunks else ""

        # Relationships where stakeholder is source
        rel_source_chunks = []
        for r in s.get("related_relationships_source", []):
            rel_source_chunks.append(
                " | ".join(
                    [
                        f"RelID:{r.get('relationship_id', '')}",
                        f"Target:{r.get('target', '')}",
                        f"Type:{r.get('relationship_type', '')}",
                        f"Subtype:{r.get('relationship_subtype', '')}",
                        f"Strength:{r.get('strength', '')}",
                        f"Formality:{r.get('relationship_formality', '')}",
                        f"CollabPotential:{r.get('collaboration_potential', '')}",
                        f"ConflictPotential:{r.get('conflict_potential', '')}",
                    ]
                )
            )
        rel_source_block = " || ".join(rel_source_chunks) if rel_source_chunks else ""

        # Relationships where stakeholder is target
        rel_target_chunks = []
        for r in s.get("related_relationships_target", []):
            rel_target_chunks.append(
                " | ".join(
                    [
                        f"RelID:{r.get('relationship_id', '')}",
                        f"Source:{r.get('source', '')}",
                        f"Type:{r.get('relationship_type', '')}",
                        f"Subtype:{r.get('relationship_subtype', '')}",
                        f"Strength:{r.get('strength', '')}",
                        f"Formality:{r.get('relationship_formality', '')}",
                        f"CollabPotential:{r.get('collaboration_potential', '')}",
                        f"ConflictPotential:{r.get('conflict_potential', '')}",
                    ]
                )
            )
        rel_target_block = " || ".join(rel_target_chunks) if rel_target_chunks else ""

        # Themes, with their own related pain points and challenges
        theme_chunks = []
        for t in s.get("related_themes", []):
            t_pp = "; ".join(
                f"{pp.get('id', '')}:{pp.get('label', '')}"
                for pp in t.get("related_pain_points", [])
            )
            t_challenges = "; ".join(t.get("potential_challenges", []))
            theme_chunks.append(
                " | ".join(
                    [
                        f"ThemeID:{t.get('theme_id', '')}",
                        f"ThemeName:{t.get('theme_name', '')}",
                        f"Description:{t.get('description', '')}",
                        f"Prevalence:{t.get('prevalence', '')}",
                        f"RelatedPainPoints:{t_pp}",
                        f"PotentialChallenges:{t_challenges}",
                    ]
                )
            )
        themes_block = " || ".join(theme_chunks) if theme_chunks else ""

        # Theme clusters
        cluster_chunks = []
        for c in s.get("related_theme_clusters", []):
            cluster_chunks.append(
                " | ".join(
                    [
                        f"ClusterName:{c.get('cluster_name', '')}",
                        f"ClusterDescription:{c.get('cluster_description', '')}",
                        f"MatchingThemes:{', '.join(c.get('matching_themes', []))}",
                    ]
                )
            )
        clusters_block = " || ".join(cluster_chunks) if cluster_chunks else ""

        parts = [core]
        if pains_block:
            parts.append(f"PainPoints:[{pains_block}]")
        if rel_source_block:
            parts.append(f"RelationshipsSource:[{rel_source_block}]")
        if rel_target_block:
            parts.append(f"RelationshipsTarget:[{rel_target_block}]")
        if themes_block:
            parts.append(f"Themes:[{themes_block}]")
        if clusters_block:
            parts.append(f"ThemeClusters:[{clusters_block}]")

        summaries.append(" ||| ".join(parts))

    return summaries


def build_level1_summaries(assignments: List[Dict]) -> List[str]:
    """Consolidate L1 assignments for L2 clustering"""
    groups = defaultdict(
        lambda: {
            "label": None,
            "members": [],
            "reasonings": [],  # ALL reasonings
        }
    )

    for a in assignments:
        cid = a["cluster_id"]
        groups[cid]["members"].append(a["stakeholder"])
        groups[cid]["reasonings"].append(a["reasoning"])
        if not groups[cid]["label"]:  # First sets label
            groups[cid]["label"] = a["cluster_label"]

    # Build summaries
    summaries = []
    for cid, data in sorted(groups.items(), key=lambda x: x[0]):
        cluster_json = {
            "id": cid,
            "label": data["label"],
            "size": len(data["members"]),
            "members": data["members"],
            "all_reasonings": "; ".join(set(data["reasonings"])),
        }
        summaries.append(
            f"Level1_Cluster{cid}: {json.dumps(cluster_json, ensure_ascii=False)}"
        )

    return summaries


# LEVEL 1 CLUSTERING
async def assign_level1_clusters(
    stakeholders: List[Dict], config: ClusteringConfig
) -> tuple[List[Dict[str, Any]], List[Dict], int]:
    """LLM decides optimal clusters in range."""
    logger.info(
        f"Level1: {len(stakeholders)} stakeholders, range {config.level1_min}-{config.level1_max}"
    )

    summaries = build_stakeholder_summaries(stakeholders)

    prompt_text = f"""INTELLIGENT CLUSTERING: Analyze {len(stakeholders)} stakeholders.

OBJECTIVE:
Perform a mutually exclusive, collectively exhaustive (MECE) clustering of the provided stakeholders.

STAKEHOLDERS SUMMARY:
{chr(10).join(summaries)}

TASK (CRITICAL RULES):
1. Review all summaries: Group by shared category, role, pain points, themes, relationships...
2. n_clusters: Decide the optimal number between {config.level1_min} and {config.level1_max}, based on natural groupings.
3. CREATE NON-OVERLAPPING groups: Assign EVERY stakeholder EXACTLY ONCE → cluster_id 0-(n_clusters-1)
4. Exhaustive: You must output EXACTLY {len(stakeholders)} assignment records. DO NOT omit any.
5. cluster_label: 1-3 words describing group theme.

Output STRICTLY VALID JSON matching schema. NO OTHER TEXT."""

    dynamic_schema = build_level1_schema(config)
    initial_state = InputState(topic=prompt_text, extraction_schema=dynamic_schema)
    llm_config = Configuration(
        model=config.model,
        prompt="Execute clustering:\n{topic}\n\nSchema: {schema}\n\nJSON ONLY - VALID JSON REQUIRED.",
        max_loops=config.max_loops,
        temperature=0.1,  # Slight creativity for labels
    ).__dict__

    final_state = await safe_llm_invoke(initial_state, llm_config)

    #  Parsing
    raw_response = final_state.get("answer", "{}")
    try:
        # Try parse_json_response first
        response_data = parse_json_response(raw_response)
    except:
        # Fallback: manual JSON parsing
        response_data = json.loads(raw_response) if raw_response.startswith("{") else {}

    #  Extraction
    n_clusters = config.level1_min  # Safe default
    assignments = []

    if isinstance(response_data, dict):
        n_clusters = response_data.get("n_clusters", config.level1_min)
        assignments = response_data.get("assignments", [])
        logger.info(
            f" Parsed dict: n={n_clusters} clusters, assigns={len(assignments)}"
        )
    elif isinstance(response_data, list):
        # LLM returned just assignments array
        assignments = response_data
        # Infer n_clusters from max cluster_id +1
        if assignments:
            max_id = max(
                a.get("cluster_id", -1) for a in assignments if isinstance(a, dict)
            )
            n_clusters = max_id + 1 if max_id >= 0 else config.level1_min
        logger.warning(
            f" List format: inferred n={n_clusters} clusters, assigns={len(assignments)}"
        )
    else:
        logger.error(f" Invalid response type: {type(response_data)}")

    # Clamp to range
    n_clusters = max(config.level1_min, min(config.level1_max, n_clusters))

    # Robust mapping
    def normalize_key(name: str) -> str:
        return name.strip("\"'").lower()

    assignment_map = {normalize_key(a["stakeholder"]): a for a in assignments}

    results = []
    for stakeholder in stakeholders:
        norm_key = normalize_key(stakeholder["stakeholder_name"])
        assignment = assignment_map.get(
            norm_key,
            {
                "cluster_id": -1,
                "cluster_label": "Unassigned",
                "reasoning": "Not assigned by LLM",
            },
        )
        enriched = {
            **stakeholder,
            "level1_cluster_id": assignment["cluster_id"],
            "level1_cluster_label": assignment["cluster_label"],
            "level1_reasoning": assignment.get("reasoning", ""),
        }
        results.append(enriched)  # Update cluster assignment to stakeholder

    return results, assignments, n_clusters


# LEVEL 2 CLUSTERING
async def assign_level2_clusters(
    level1_results: List[Dict],
    level1_assignments: List[Dict],
    config: ClusteringConfig,
) -> List[Dict]:
    """LLM groups level1 clusters."""
    logger.info(
        f"Level2: {len(set(a['cluster_id'] for a in level1_assignments))} groups, range {config.level2_min}-{config.level2_max}"
    )

    group_summaries = build_level1_summaries(level1_assignments)

    prompt_text = f"""STRATEGIC SUPER-CLUSTERING: Higher-Level Ecosystem Synthesis.

OBJECTIVE:
Aggregate the {len(group_summaries)} Level 1 clusters into broader, strategic 'Super-Clusters' (Macro-Domains).

LEVEL1 GROUPS:
{chr(10).join(group_summaries)}

TASK:
1. Synthesize the Level 1 groups into a coherent high-level ecosystem map.
2. n_super_clusters: Choose between {config.level2_min} and {config.level2_max} based on macro-alignment.
3. Hard Partition: Every Level 1 Cluster must be mapped to exactly ONE Super-Cluster.
4. super_cluster_label: Descriptive theme (2-4 words)

Output STRICT JSON matching schema."""

    dynamic_schema = build_level2_schema(config)
    initial_state = InputState(topic=prompt_text, extraction_schema=dynamic_schema)
    llm_config = Configuration(
        model=config.model,
        prompt="Super-cluster:\n{topic}\nSchema: {schema}\nJSON ONLY.",
        max_loops=config.max_loops,
        temperature=0.1,
    ).__dict__

    final_state = await safe_llm_invoke(initial_state, llm_config)

    raw_response = final_state.get("answer", "{}")

    # Parse with fallback
    try:
        response_data = parse_json_response(raw_response)
    except Exception as parse_err:
        logger.warning(f"parse_json_response failed: {parse_err}, using json.loads")
        try:
            response_data = (
                json.loads(raw_response) if isinstance(raw_response, str) else {}
            )
        except:
            response_data = {}

    #  Extraction
    n_super = config.level2_min  # Safe default
    super_assignments = []

    if isinstance(response_data, dict):
        n_super = response_data.get("n_super_clusters", config.level2_min)
        super_assignments = response_data.get("super_assignments", [])
        logger.info(
            f"Level2 dict: n_super={n_super} clusters, assigns={len(super_assignments)}"
        )

    elif isinstance(response_data, list):
        super_assignments = [item for item in response_data if isinstance(item, dict)]
        # Infer n_super from max super_cluster_id
        if super_assignments:
            cluster_ids = [a.get("super_cluster_id", -1) for a in super_assignments]
            max_id = max(cluster_ids)
            n_super = max_id + 1 if max_id >= 0 else config.level2_min
        logger.info(
            f"Level2 list: inferred n_super={n_super} clusters, assigns={len(super_assignments)}"
        )

    else:
        logger.warning(f"Level2 unexpected format: {type(response_data)}")

    # Map back via cluster_label
    super_map = {sa["level1_cluster"].strip(): sa for sa in super_assignments}

    for rec in level1_results:
        l1_label = rec.get("level1_cluster_label", "").strip()
        if l1_label in super_map:
            sa = super_map[l1_label]
            rec.update(
                {
                    "level2_cluster_id": sa["super_cluster_id"],
                    "level2_cluster_label": sa["super_cluster_label"],
                    "level2_reasoning": sa.get("reasoning", ""),
                }
            )
        else:
            rec.update(
                {
                    "level2_cluster_id": -1,
                    "level2_cluster_label": "Unassigned",
                    "level2_reasoning": "Level1 group not super-clustered",
                }
            )

    return level1_results


# MAIN PIPELINE: 2 LLM CALLS
async def assign_stakeholder_clusters(
    ecosystem_data: Dict, config: ClusteringConfig
) -> List[Dict]:
    """Clustering pipeline"""
    logger.info("=== STARTING CLUSTERING PIPELINE ===")
    stakeholder_records = flatten_ecosystem_to_stakeholder_records(ecosystem_data)
    logger.info(f"Flattened: {len(stakeholder_records)} stakeholders")

    level1_results, level1_assign, l1_n = await assign_level1_clusters(
        stakeholder_records, config
    )
    final_results = await assign_level2_clusters(level1_results, level1_assign, config)

    logger.info(
        f"COMPLETE: L1={l1_n} clusters → L2={len(set(r['level2_cluster_id'] for r in final_results if r['level2_cluster_id'] >= 0))} super-clusters"
    )
    return final_results


# CLI USAGE
async def main():
    parser = argparse.ArgumentParser(description="Adaptive Stakeholder Clustering")
    parser.add_argument("--input", "-i", required=True, help="Ecosystem JSON file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--level1-min", type=int, default=6, help="Min level1 clusters")
    parser.add_argument(
        "--level1-max", type=int, default=10, help="Max level1 clusters"
    )
    parser.add_argument(
        "--level2-min", type=int, default=2, help="Min level2 super-clusters"
    )
    parser.add_argument(
        "--level2-max", type=int, default=5, help="Max level2 super-clusters"
    )
    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Input error: {e}")
        return

    config = ClusteringConfig(
        output_dir=args.output_dir,
        level1_min=args.level1_min,
        level1_max=args.level1_max,
        level2_min=args.level2_min,
        level2_max=args.level2_max,
    )

    clustered = await assign_stakeholder_clusters(input_data, config)

    input_filename = Path(args.input).name
    output_filename = input_filename.replace(".json", "_clusters.json")
    save_output(clustered, output_filename, args.output_dir)

    summary_df = pd.DataFrame(clustered)[
        [
            "stakeholder_name",
            "level1_cluster_id",
            "level1_cluster_label",
            "level2_cluster_id",
            "level2_cluster_label",
        ]
    ]

    total_stakeholders = len(clustered)
    l1_clusters = len(
        set(r["level1_cluster_id"] for r in clustered if r["level1_cluster_id"] >= 0)
    )
    l2_clusters = len(
        set(r["level2_cluster_id"] for r in clustered if r["level2_cluster_id"] >= 0)
    )

    logger.info(f"""
     CLUSTERING SUMMARY
    ===================
    Total Stakeholders: {total_stakeholders}
    Level 1 Clusters:   {l1_clusters} 
    Level 2 Clusters:   {l2_clusters}

    CLUSTER BREAKDOWN:
    {summary_df.groupby(["level2_cluster_label", "level1_cluster_label"]).size().sort_index().to_string()}

    PREVIEW:
    {summary_df.head(10).to_string(index=False)}
    """)

    # Save summary CSV too
    summary_df.to_csv(Path(args.output_dir) / "cluster_summary.csv", index=False)
    logger.info(" Saved cluster_summary.csv")


if __name__ == "__main__":
    import pandas as pd  # For summary

    asyncio.run(main())
