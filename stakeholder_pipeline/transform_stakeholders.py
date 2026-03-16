"""Helper to consolidate stakeholder data.
Used in dynamic heirarchy pipeline and macromicro pipeline
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from stakeholder_pipeline.utils import save_output


def transform_stakeholder_data(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform stakeholder data, extracting ALL roles from all_sources array."""
    consolidated = input_data.get("consolidated_stakeholders", [])
    relationships = input_data.get("relationships", [])
    pain_points = input_data.get("pain_points", [])

    # Group by canonical_name -> {category: [all_role_descriptions]}
    roles_by_stakeholder = defaultdict(lambda: defaultdict(list))

    for stakeholder in consolidated:
        canonical = stakeholder["Canonical Name"]
        category = stakeholder["Category"]

        # Extract from primary Role
        roles_by_stakeholder[canonical][category].append(stakeholder["Role"])

        # Extract ALL roles from all_sources if present
        if (
            "consolidation_info" in stakeholder
            and "all_sources" in stakeholder["consolidation_info"]
        ):
            for source in stakeholder["consolidation_info"]["all_sources"]:
                if "Role" in source:
                    roles_by_stakeholder[canonical][category].append(source["Role"])

        # Deduplicate roles per category (keep unique)
        for cat in roles_by_stakeholder[canonical]:
            roles_by_stakeholder[canonical][cat] = list(
                set(roles_by_stakeholder[canonical][cat])
            )

    # Convert lists back to dicts with concatenated descriptions
    formatted_roles = {}
    for canonical, categories in roles_by_stakeholder.items():
        formatted_roles[canonical] = {}
        for category, roles_list in categories.items():
            # Join multiple roles with semicolon separator
            formatted_roles[canonical][category] = "; ".join(roles_list)

    # Relationships (unchanged)
    rels_by_source = defaultdict(list)
    for rel in relationships:
        source = rel["source"]
        target = rel["target"]
        rel_type = rel["relationship_category"]
        desc = rel["relationship_description"]
        rels_by_source[source].append(
            {"target": target, "relationship_type": rel_type, "description": desc}
        )

    # Pain points (unchanged)
    pains_by_stakeholder = defaultdict(list)
    for pain in pain_points:
        pains_by_stakeholder[pain["stakeholder"]].append(
            {"category": pain["painpoint_category"], "description": pain["painpoint"]}
        )

    # All unique stakeholders
    all_stakeholders = set(formatted_roles.keys())
    for source in rels_by_source:
        all_stakeholders.add(source)
        for rel in rels_by_source[source]:
            all_stakeholders.add(rel["target"])
    for stakeholder in pains_by_stakeholder:
        all_stakeholders.add(stakeholder)

    # Build result
    result = []
    for canonical in sorted(all_stakeholders):
        result.append(
            {
                "canonical_name": canonical,
                "roles": formatted_roles.get(canonical, {}),
                "relationships": rels_by_source[canonical],
                "painpoints": pains_by_stakeholder[canonical],
            }
        )

    return result


# Usage example
if __name__ == "__main__":
    # Your original JSON data here
    input_file = "output/test_policy_output_relationship.json"

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    transformed = transform_stakeholder_data(data)

    input_path = Path(input_file).name
    output_filename = input_path.replace("_relationship.json", "_transformed.json")
    save_output(
        result=transformed,
        output_filename=output_filename,
        output_dir="output",
    )
