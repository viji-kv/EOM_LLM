import json
from typing import Any, Dict, List


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
            # "sources": sh["sources"],
            # Initialize related data lists
            "related_pain_points": [],
            # "related_pain_points_affected": [],
            # "related_pain_points_causing": [],
            "related_relationships_source": [],
            "related_relationships_target": [],
            "related_themes": [],
            "related_theme_clusters": [],
        }

    # Attach pain point info (both affected and causing) in the flat schema you showed
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
                    # "pain_point_relation_type": "affected",
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


########################

input = "output/5.2 v5_2026_03_11.json"
# input_path = project_output / input
# input = "output/test_policy_output.json"
with open(input, encoding="utf-8") as f:
    ecosystem_data = json.load(f)

flattened_records = flatten_ecosystem_to_stakeholder_records(ecosystem_data)
