######################################################
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

from numpy import single

input_file = "output/stakeholders_output_AP HK - Financial industry.json"


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
            # "confidence": {
            #     "type": "string",
            #     "enum": ["High", "Medium", "Low"],
            #     "description": "Confidence in this clustering decision",
            # },
            # "reasoning": {
            #     "type": "string",
            #     "description": "Why these were clustered together",
            # },
        },
        "required": ["cluster_id", "canonical_name", "member_indices"],
    },
}

from enrichment.state import InputState
from enrichment.configuration import Configuration
from enrichment import graph


def parse_json_response(raw_info: str) -> List[Dict]:
    """Robust JSON parser for LLM responses with multiple extraction strategies."""
    if not raw_info:
        print("      ⚠️  Empty raw response")
        return []

    print(f"      🔍 Raw response preview: {repr(raw_info[:300])}...")

    # Strategy 1: Extract between ```json ... ```
    json_match = re.search(
        r"```json?\s*(\[.*?\])\s*```", raw_info, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        json_str = json_match.group(1)
        print("      ✅ Strategy 1: Found ```json block")
    else:
        # Strategy 2: Extract largest JSON array candidate
        array_match = re.search(
            r"\[\s*(?:\{[^}]*\}|\d+|\[[^\]]*\]|\s*,\s*)*\s*\]", raw_info, re.DOTALL
        )
        if array_match:
            json_str = array_match.group(0)
            print("      ✅ Strategy 2: Found array candidate")
        else:
            print("      ❌ No JSON block or array found")
            return []

    # Clean the extracted JSON
    json_str = re.sub(r"\\n", "", json_str)  # Remove escaped newlines
    json_str = re.sub(r"\\", "", json_str)  # Remove backslashes
    json_str = json_str.strip()

    if len(json_str) < 10:
        print("      ⚠️  Extracted JSON too short")
        return []

    print(f"      🔍 Cleaned JSON preview: {repr(json_str[:200])}...")

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            print(f"      ✅ Successfully parsed {len(parsed)} clusters")
            return parsed
        elif isinstance(parsed, dict):
            print("      ⚠️  Parsed single dict, wrapping in list")
            return [parsed]
        else:
            print(f"      ⚠️  Unexpected type: {type(parsed)}")
            return []
    except json.JSONDecodeError as e:
        print(f"      ❌ JSONDecodeError: {e}")
        print(
            f"      🔍 Error position preview: {repr(json_str[max(0, e.pos - 50) : e.pos + 50])}"
        )
        return []
    except Exception as e:
        print(f"      ❌ Unexpected error: {type(e).__name__}: {e}")
        return []


async def llm_cluster_stakeholders(
    stakeholders: List[Dict],
    category: str,
    # config: LLMClusterConfig
) -> List[Dict]:
    """
    Use LLM to cluster stakeholders within a category.
    WARNING: Expensive and can be inconsistent.
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
        prompt_text += f"{i}. {name} - {role} (conf: {conf})\n"

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

    llm_config = Configuration(
        model="openai/gpt-4o-mini",
        prompt="Cluster stakeholders:\n{topic}\nSchema: {schema}\nOutput valid JSON only.",
        max_loops=2,
    ).__dict__

    print(f"      🤖 LLM clustering {len(stakeholders)} {category} stakeholders...")

    final_state = await graph.ainvoke(initial_state, llm_config)
    print(f"Final state: {final_state}")
    print(final_state.get("answer", "No answer from LLM"))
    # # Parse response
    # try:
    #     result = json.loads(
    #         re.sub(r"```json?|\n*```", "", final_state.get("answer", "").strip())
    #     )
    #     return result if isinstance(result, list) else []
    # except Exception as e:
    #     print(f"      ❌ LLM parsing error: {e}")
    #     return []
    return parse_json_response(final_state.get("answer", ""))


def merge_llm_clusters(
    clusters: List[Dict],
    stakeholders: List[Dict],
    original_indices: List[int],  # 🔴 CHANGE: Pass actual indices for this category
) -> List[Dict]:
    """
    Merge stakeholders based on LLM clustering with STRICT validation.
    """
    consolidated = []
    used_indices = set()

    # 🔴 CHANGE: Create mapping from LLM indices (0,1,2...) to real indices
    index_map = {i: real_idx for i, real_idx in enumerate(original_indices)}
    max_llm_index = len(original_indices) - 1
    print(f"index_map: {index_map}")

    print(f"      Validating LLM clusters (max index: {max_llm_index})...")

    for cluster_info in clusters:
        member_indices = cluster_info.get("member_indices", [])
        print(f"member_indices:{member_indices}")
        if not member_indices:
            continue

        # 🔴 CHANGE: Validate ALL indices before processing
        valid_llm_indices = []
        invalid_count = 0

        for llm_idx in member_indices:
            if not isinstance(llm_idx, int):
                invalid_count += 1
                continue
            if llm_idx < 0 or llm_idx > max_llm_index:
                print(
                    f"      ⚠️  LLM hallucinated index {llm_idx} (max is {max_llm_index})"
                )
                invalid_count += 1
                continue
            if llm_idx in valid_llm_indices:
                print(f"      ⚠️  Duplicate index {llm_idx} in cluster")
                continue
            valid_llm_indices.append(llm_idx)

        if invalid_count > 0:
            print(
                f"      ⚠️  Cluster {cluster_info.get('cluster_id')} had {invalid_count} invalid indices"
            )

        if not valid_llm_indices:
            continue

        # 🔴 CHANGE: Map to real stakeholder indices
        real_indices = [index_map[i] for i in valid_llm_indices if i in index_map]
        print(f"real_indices (before filtering used): {real_indices}")
        # 🔴 CHANGE: Check for already used indices (LLM might duplicate)
        real_indices = [i for i in real_indices if i not in used_indices]

        if not real_indices:
            print(
                f"      ⚠️  Cluster {cluster_info.get('cluster_id')} has no valid/unused indices"
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
            # "method": "llm_only_clustering",
            "member_indices": real_indices,
            # "llm_reasoning": cluster_info.get("reasoning", ""),
            # "llm_confidence": cluster_info.get("confidence", "Unknown"),
            "all_sources": all_sources,
        }

        consolidated.append(master)

    # 🔴 CHANGE: Add ONLY unused stakeholders from original_indices
    for real_idx in original_indices:
        if real_idx not in used_indices:
            singleton = stakeholders[real_idx].copy()
            singleton["consolidation_info"] = {
                "cluster_size": 1,
                # "method": "llm_singleton",
                "member_indices": [real_idx],
            }
            consolidated.append(singleton)
            used_indices.add(real_idx)

    print(
        f"      ✅ Created {len(consolidated)} consolidated (from {len(original_indices)} original)"
    )

    return consolidated


###########################################
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

stk_list = []
stakeholders = data.get("stakeholders", [])
for i, s in enumerate(stakeholders):
    name = s.get("Canonical Name", "Unknown")
    role = s.get("Role", "Unknown")
    stk_list.append(f"{i}. {name} - {role}")

print(f"Consolidating {len(stakeholders)} stakeholders")

category_blocks = defaultdict(list)

for i, s in enumerate(stakeholders):
    cat = s.get("Category", "Unknown")
    category_blocks[cat].append(i)

print(f"   Created {len(category_blocks)} category blocks")

partners_only = {"Partner": category_blocks["Partner"]}
print(partners_only)

all_consolidated = []
for cat, indices in category_blocks.items():
    cat_stakeholders = [stakeholders[i] for i in indices]
    # stkholder_consolidated = []
    if len(cat_stakeholders) > 20:
        batch_size = 5
        # batch_consolidated = []
        for i in range(0, len(cat_stakeholders), batch_size):
            batch = cat_stakeholders[i : i + batch_size]
            original_indices = indices[i : i + batch_size]
            # print(indices[i : i + batch_size])
            response = await llm_cluster_stakeholders(batch, cat)
            print(f"RESPONSE:{response} \n\n")
            print(each["Canonical Name"] for each in batch)
            # consolidated = []
            # used_indices = set()
            # print(f"consolidated:{consolidated}")
            # print(f"used_indices:{used_indices}")
            # # 🔴 CHANGE: Create mapping from LLM indices (0,1,2...) to real indices
            # index_map = {i: real_idx for i, real_idx in enumerate(original_indices)}
            # max_llm_index = len(original_indices) - 1
            # print(f"index_map:{index_map}")
            # print(f"      Validating LLM clusters (max index: {max_llm_index})...")

            # for cluster_info in response:
            #     member_indices = cluster_info.get("member_indices", [])
            #     print(f"member_indices:{member_indices}")
            #     if not member_indices:
            #         continue

            #     # 🔴 CHANGE: Validate ALL indices before processing
            #     valid_llm_indices = []
            #     invalid_count = 0

            #     for llm_idx in member_indices:
            #         if not isinstance(llm_idx, int):
            #             invalid_count += 1
            #             continue
            #         if llm_idx < 0 or llm_idx > max_llm_index:
            #             print(
            #                 f"      ⚠️  LLM hallucinated index {llm_idx} (max is {max_llm_index})"
            #             )
            #             invalid_count += 1
            #             continue
            #         if llm_idx in valid_llm_indices:
            #             print(f"      ⚠️  Duplicate index {llm_idx} in cluster")
            #             continue
            #         valid_llm_indices.append(llm_idx)
            #     print(f"valid_llm_indices:{valid_llm_indices}")
            #     if invalid_count > 0:
            #         print(
            #             f"      ⚠️  Cluster {cluster_info.get('cluster_id')} had {invalid_count} invalid indices"
            #         )

            #     if not valid_llm_indices:
            #         continue

            #     # 🔴 CHANGE: Map to real stakeholder indices
            #     real_indices = [
            #         index_map[i] for i in valid_llm_indices if i in index_map
            #     ]
            #     print(f"real_indices:{real_indices}")
            #     # 🔴 CHANGE: Check for already used indices (LLM might duplicate)
            #     real_indices = [i for i in real_indices if i not in used_indices]
            #     print(f"valid_llm_indices:{valid_llm_indices}")
            #     print(f"real_indices:{real_indices}")
            #     if not real_indices:
            #         print(
            #             f"      ⚠️  Cluster {cluster_info.get('cluster_id')} has no valid/unused indices"
            #         )
            #         continue
            #     print(f"used_indices:{used_indices}")
            #     used_indices.update(real_indices)
            #     print(f"used_indices:{used_indices}")
            #     # Pick highest confidence as primary
            #     scores = [
            #         (int(stakeholders[i].get("Confidence Score", "0").rstrip("%")), i)
            #         for i in real_indices
            #     ]
            #     primary_idx = max(scores)[1]
            #     print(f"primary_idx:{primary_idx}")
            #     master = stakeholders[primary_idx].copy()
            #     print(f"master:{master}")
            #     master["Canonical Name"] = cluster_info.get(
            #         "canonical_name", master.get("Canonical Name")
            #     )
            #     print(f"master:{master}")

            #     # Aggregate sources
            #     all_sources = []
            #     for i in real_indices:
            #         meta = stakeholders[i].get("Source metadata", {})
            #         all_sources.append(
            #             {
            #                 "index": i,
            #                 "original_name": stakeholders[i].get("Stakeholder Name"),
            #                 "filename": meta.get("filename"),
            #                 "confidence": stakeholders[i].get("Confidence Score"),
            #             }
            #         )

            #     master["consolidation_info"] = {
            #         "cluster_size": len(real_indices),
            #         # "method": "llm_only_clustering",
            #         "member_indices": real_indices,
            #         # "llm_reasoning": cluster_info.get("reasoning", ""),
            #         # "llm_confidence": cluster_info.get("confidence", "Unknown"),
            #         "all_sources": all_sources,
            #     }

            #     consolidated.append(master)
            #     print(f"consolidated:{consolidated}")

            # # 🔴 CHANGE: Add ONLY unused stakeholders from original_indices
            # print(f"original_indices:{original_indices}")
            # print(f"used_indices:{used_indices}")

            # for real_idx in original_indices:
            #     if real_idx not in used_indices:
            #         singleton = stakeholders[real_idx].copy()
            #         singleton["consolidation_info"] = {
            #             "cluster_size": 1,
            #             # "method": "llm_singleton",
            #             "member_indices": [real_idx],
            #         }
            #         consolidated.append(singleton)
            #         used_indices.add(real_idx)
            #         print(f"consolidated:{consolidated}")
            #         print(f"used_indices:{used_indices}")
            # print(
            #     f"      ✅ Created {len(consolidated)} consolidated (from {len(original_indices)} original)"
            # )

            merged = merge_llm_clusters(response, stakeholders, original_indices)
            all_consolidated.extend(merged)

    else:
        # Process whole category (no batching needed)
        clusters = await llm_cluster_stakeholders(cat_stakeholders, cat)

        #         consolidated = []
        #         used_indices = set()
        #         print(f"consolidated:{consolidated}")
        #         print(f"used_indices:{used_indices}")
        #         # 🔴 CHANGE: Create mapping from LLM indices (0,1,2...) to real indices
        #         index_map = {i: real_idx for i, real_idx in enumerate(original_indices)}
        #         max_llm_index = len(original_indices) - 1
        #         print(f"index_map:{index_map}")
        #         print(f"      Validating LLM clusters (max index: {max_llm_index})...")

        #         for cluster_info in response:
        #             member_indices = cluster_info.get("member_indices", [])
        #             print(f"member_indices:{member_indices}")
        #             if not member_indices:
        #                 continue

        #             # 🔴 CHANGE: Validate ALL indices before processing
        #             valid_llm_indices = []
        #             invalid_count = 0

        #             for llm_idx in member_indices:
        #                 if not isinstance(llm_idx, int):
        #                     invalid_count += 1
        #                     continue
        #                 if llm_idx < 0 or llm_idx > max_llm_index:
        #                     print(
        #                         f"      ⚠️  LLM hallucinated index {llm_idx} (max is {max_llm_index})"
        #                     )
        #                     invalid_count += 1
        #                     continue
        #                 if llm_idx in valid_llm_indices:
        #                     print(f"      ⚠️  Duplicate index {llm_idx} in cluster")
        #                     continue
        #                 valid_llm_indices.append(llm_idx)
        #             print(f"valid_llm_indices:{valid_llm_indices}")
        #             if invalid_count > 0:
        #                 print(
        #                     f"      ⚠️  Cluster {cluster_info.get('cluster_id')} had {invalid_count} invalid indices"
        #                 )

        #             if not valid_llm_indices:
        #                 continue

        #             # 🔴 CHANGE: Map to real stakeholder indices
        #             real_indices = [index_map[i] for i in valid_llm_indices if i in index_map]
        #             print(f"real_indices:{real_indices}")
        #             # 🔴 CHANGE: Check for already used indices (LLM might duplicate)
        #             real_indices = [i for i in real_indices if i not in used_indices]
        #             print(f"valid_llm_indices:{valid_llm_indices}")
        #             print(f"real_indices:{real_indices}")
        #             if not real_indices:
        #                 print(
        #                     f"      ⚠️  Cluster {cluster_info.get('cluster_id')} has no valid/unused indices"
        #                 )
        #                 continue
        #             print(f"used_indices:{used_indices}")
        #             used_indices.update(real_indices)
        #             print(f"used_indices:{used_indices}")
        #             # Pick highest confidence as primary
        #             scores = [
        #                 (int(stakeholders[i].get("Confidence Score", "0").rstrip("%")), i)
        #                 for i in real_indices
        #             ]
        #             primary_idx = max(scores)[1]
        #             print(f"primary_idx:{primary_idx}")
        #             master = stakeholders[primary_idx].copy()
        #             print(f"master:{master}")
        #             master["Canonical Name"] = cluster_info.get(
        #                 "canonical_name", master.get("Canonical Name")
        #             )
        #             print(f"master:{master}")

        #             # Aggregate sources
        #             all_sources = []
        #             for i in real_indices:
        #                 meta = stakeholders[i].get("Source metadata", {})
        #                 all_sources.append(
        #                     {
        #                         "index": i,
        #                         "original_name": stakeholders[i].get("Stakeholder Name"),
        #                         "filename": meta.get("filename"),
        #                         "confidence": stakeholders[i].get("Confidence Score"),
        #                     }
        #                 )

        #             master["consolidation_info"] = {
        #                 "cluster_size": len(real_indices),
        #                 # "method": "llm_only_clustering",
        #                 "member_indices": real_indices,
        #                 # "llm_reasoning": cluster_info.get("reasoning", ""),
        #                 # "llm_confidence": cluster_info.get("confidence", "Unknown"),
        #                 "all_sources": all_sources,
        #             }

        #             consolidated.append(master)
        #             print(f"consolidated:{consolidated}")

        #         # 🔴 CHANGE: Add ONLY unused stakeholders from original_indices
        #         print(f"original_indices:{original_indices}")
        #         print(f"used_indices:{used_indices}")

        #         for real_idx in original_indices:
        #             if real_idx not in used_indices:
        #                 singleton = stakeholders[real_idx].copy()
        #                 singleton["consolidation_info"] = {
        #                     "cluster_size": 1,
        #                     # "method": "llm_singleton",
        #                     "member_indices": [real_idx],
        #                 }
        #                 consolidated.append(singleton)
        #                 used_indices.add(real_idx)
        #                 print(f"consolidated:{consolidated}")
        #                 print(f"used_indices:{used_indices}")
        #         print(
        #             f"      ✅ Created {len(consolidated)} consolidated (from {len(original_indices)} original)"
        #         )

        # Pass full indices list for this category
        merged = merge_llm_clusters(clusters, stakeholders, indices)
        all_consolidated.extend(merged)

len(all_consolidated)
