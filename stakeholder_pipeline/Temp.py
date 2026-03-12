# import json
# from pathlib import Path

# script_dir = Path.cwd()  # Or Path(__file__).parent in .py scripts
# project_output = script_dir.parent / "output"  # sibling to stakeholder_pipeline


################

import pandas as pd


def flatten_ecosystem_data(json_input):
    data = json_input.get("ecosystem_analysis", {})

    # 1. Load basic stakeholder info into a dictionary for easy lookup
    stakeholders = {sh["name"]: sh for sh in data.get("stakeholders", [])}

    # 2. Pre-process Pain Points: Map which stakeholders are involved in which PP
    pp_map = {}
    for pp in data.get("pain_points", []):
        affected = pp.get("affected_stakeholders", [])
        causing = pp.get("causing_stakeholders", [])
        all_involved = list(set(affected + causing))

        for sh_name in all_involved:
            if sh_name not in pp_map:
                pp_map[sh_name] = []
            pp_map[sh_name].append(f"{pp['id']}: {pp['label']}")

    # 3. Pre-process Themes: Map stakeholders to themes
    theme_map = {}
    for theme in data.get("themes", []):
        related_sh = theme.get("related_stakeholders", [])
        for sh_name in related_sh:
            if sh_name not in theme_map:
                theme_map[sh_name] = []
            theme_map[sh_name].append(f"{theme['id']}: {theme['theme_name']}")

    # 4. Build the flattened list
    flattened_rows = []

    for name, info in stakeholders.items():
        row = {
            "id": info.get("id"),
            "name": name,
            "category": info.get("category"),
            "hierarchy": info.get("hierarchy_level"),
            "role": info.get("role"),
            "influence": info.get("influence_scope"),
            "intervention_capacity": info.get("intervention_capacity"),
            # Join the mapped data into strings for LLM readability
            "associated_themes": "; ".join(theme_map.get(name, ["None"])),
            "linked_pain_points": "; ".join(pp_map.get(name, ["None"])),
        }
        flattened_rows.append(row)

    return pd.DataFrame(flattened_rows)


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
            "related_pain_points_affected": [],
            "related_pain_points_causing": [],
            "related_relationships_source": [],
            "related_relationships_target": [],
            "related_themes": [],
            "related_theme_clusters": [],
        }

    # # Helper to serialise complex objects as strings, like in your sheet
    # def to_json_str(obj: Any) -> str:
    #     return json.dumps(obj, ensure_ascii=False)

    # Attach pain point info (both affected and causing) in the flat schema you showed
    for pp in pain_points:
        pp_base = {
            "pain_point_id": pp["id"],
            "label": pp["label"],
            "category_pain_point": pp["category"],
            "description_pain_point": pp["description"],
            "hierarchy_level_pain_point": pp["hierarchy_level"],
            # "confidence_pain_point": pp["confidence"],
            "severity": pp["severity"],
            "urgency": pp["urgency"],
            "intervention_difficulty": pp["intervention_difficulty"],
            # "affected_stakeholders": pp.get("affected_stakeholders", []),
            "causing_stakeholders": pp.get("causing_stakeholders", []),
            # "stakeholder_impact_json": pp.get("stakeholder_impact", {}),
            # 'pain_point_sources_json': to_json_str(pp.get('sources', [])),
        }

        impacts = pp.get("stakeholder_impact", {})

        # Affected link
        for name in pp.get("affected_stakeholders", []):
            if name in stakeholder_records:
                rec = stakeholder_records[name]
                row = {
                    **pp_base,
                    "pain_point_relation_type": "affected",
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
                    "type": rel["relationship_type"],
                    "subtype": rel["relationship_subtype"],
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
                    "type": rel["relationship_type"],
                    "subtype": rel["relationship_subtype"],
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
                        "matching_themes": matching_themes,
                    }
                )

    # Convert to list and sort by stakeholder ID
    result = list(stakeholder_records.values())
    result.sort(key=lambda x: x["stakeholder_id"])

    return result


input = "output/5.2 v5_2026_03_11.json"
# input_path = project_output / input
# input = "output/test_policy_output.json"
with open(input, encoding="utf-8") as f:
    ecosystem_data = json.load(f)

flattened_records = flatten_ecosystem_to_stakeholder_records(ecosystem_data)


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
        "related_pain_points_affected": [],
        "related_pain_points_causing": [],
        "related_relationships_source": [],
        "related_relationships_target": [],
        "related_themes": [],
        "related_theme_clusters": [],
    }


# # Helper to serialise complex objects as strings, like in your sheet
# def to_json_str(obj: Any) -> str:
#     return json.dumps(obj, ensure_ascii=False)

# Attach pain point info (both affected and causing) in the flat schema you showed
for pp in pain_points:
    pp_base = {
        "pain_point_id": pp["id"],
        "label": pp["label"],
        "category_pain_point": pp["category"],
        "description_pain_point": pp["description"],
        "hierarchy_level_pain_point": pp["hierarchy_level"],
        # "confidence_pain_point": pp["confidence"],
        "severity": pp["severity"],
        "urgency": pp["urgency"],
        "intervention_difficulty": pp["intervention_difficulty"],
        # "affected_stakeholders": pp.get("affected_stakeholders", []),
        "causing_stakeholders": pp.get("causing_stakeholders", []),
        # "stakeholder_impact_json": pp.get("stakeholder_impact", {}),
        # 'pain_point_sources_json': to_json_str(pp.get('sources', [])),
    }

    impacts = pp.get("stakeholder_impact", {})

    # Affected link
    for name in pp.get("affected_stakeholders", []):
        if name in stakeholder_records:
            rec = stakeholder_records[name]
            row = {
                **pp_base,
                "pain_point_relation_type": "affected",
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
                "type": rel["relationship_type"],
                "subtype": rel["relationship_subtype"],
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
                "type": rel["relationship_type"],
                "subtype": rel["relationship_subtype"],
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
                    "matching_themes": matching_themes,
                }
            )

# Convert to list and sort by stakeholder ID
result = list(stakeholder_records.values())


###################################################

# Example usage:
# json_str = '''{your json here}'''
# data = json.loads(json_str)
# flattened_records = flatten_ecosystem_to_stakeholder_records(data)
#
# # Save to CSV or JSON
# import pandas as pd
# df = pd.json_normalize(flattened_records)
# df.to_csv('stakeholder_records.csv', index=False)
# print(f"Flattened {len(flattened_records)} stakeholder records")


########################################

# Example Usage:
# with open('data.json', 'r') as f:
#    raw_json = json.load(f)
# df = flatten_ecosystem_data(raw_json)
# print(df.to_string())

import json

input = "output/5.2 v5_2026_03_11.json"
# input_path = project_output / input
# input = "output/test_policy_output.json"
with open(input, encoding="utf-8") as f:
    input_data = json.load(f)

# df = flatten_ecosystem_data(input_data)
# print(df.to_string())

# input_data["current_analysis_result"]["ecosystem_analysis"].keys()


data = input_data.get("current_analysis_result", {}).get("ecosystem_analysis", {})
data.keys()
# 1. Load basic stakeholder info into a dictionary for easy lookup
stakeholders = {sh["name"]: sh for sh in data.get("stakeholders", [])}

# 2. Pre-process Pain Points: Map which stakeholders are involved in which PP
pp_map = {}
for pp in data.get("pain_points", []):
    affected = pp.get("affected_stakeholders", [])
    causing = pp.get("causing_stakeholders", [])
    all_involved = list(set(affected + causing))

    for sh_name in all_involved:
        if sh_name not in pp_map:
            pp_map[sh_name] = []
        pp_map[sh_name].append(f"{pp['id']}: {pp['label']}")

# 3. Pre-process Themes: Map stakeholders to themes
theme_map = {}
for theme in data.get("themes", []):
    related_sh = theme.get("related_stakeholders", [])
    for sh_name in related_sh:
        if sh_name not in theme_map:
            theme_map[sh_name] = []
        theme_map[sh_name].append(f"{theme['id']}: {theme['theme_name']}")

# 4. Build the flattened list
flattened_rows = []

for name, info in stakeholders.items():
    row = {
        "id": info.get("id"),
        "name": name,
        "category": info.get("category"),
        "hierarchy": info.get("hierarchy_level"),
        "role": info.get("role"),
        "influence": info.get("influence_scope"),
        "intervention_capacity": info.get("intervention_capacity"),
        # Join the mapped data into strings for LLM readability
        "associated_themes": "; ".join(theme_map.get(name, ["None"])),
        "linked_pain_points": "; ".join(pp_map.get(name, ["None"])),
    }
    flattened_rows.append(row)
