import re
from rapidfuzz import fuzz  # string matching
from typing import List, Dict
from collections import defaultdict


def merge_stakeholders(stakeholders: List[Dict], threshold: float = 82) -> List[Dict]:
    """Merge duplicates"""

    def normalize_match_key(name: str) -> str:
        """normalization"""

        govt_noise = r"\b(govt?|government|ministry|department|agency|authority|bureau|commission)\b"
        name = re.sub(govt_noise, "", name, flags=re.I)

        # Synonyms (preserve locations)
        name = re.sub(r"\b(elderly|senior|old|aged)\b", "elderly", name, flags=re.I)
        name = re.sub(
            r"\b(community|association|group|network)\b", "community", name, flags=re.I
        )

        # Clean
        name = re.sub(r"\s+", " ", name.lower().strip())
        return name.strip()

    merged = []
    used = set()  # index of already clustered records

    for i, s1 in enumerate(stakeholders):
        # print(i, s1)
        if i in used:
            continue

        s1_key = normalize_match_key(s1["Stakeholder Name"])
        print(s1_key)
        cluster = [
            s1
        ]  # keeps similar records for merging [{'Stakeholder Name':..},{'Stakeholder Name':..}]

        for j, s2 in enumerate(stakeholders):
            if i == j or j in used:
                continue

            s2_key = normalize_match_key(s2["Stakeholder Name"])

            # Higher threshold for safety
            if fuzz.partial_ratio(s1_key, s2_key) >= threshold:
                cluster.append(s2)
                used.add(j)

        # Merge logic (unchanged)
        best = max(cluster, key=lambda x: float(x["Confidence Score"].strip("%")))
        merged_record = best.copy()
        merged_record["Canonical Name"] = best["Stakeholder Name"]
        merged_record["Cluster Size"] = len(cluster)
        merged_record["Aliases"] = [s["Stakeholder Name"] for s in cluster if s != best]
        merged_record["Match Keys"] = [
            normalize_match_key(s["Stakeholder Name"]) for s in cluster
        ]

        merged.append(merged_record)
        used.add(i)

    return merged


def main():
    # TEST CASES
    stakeholders = [
        {
            "Stakeholder Name": "Taiwan Government",
            "Confidence Score": "95%",
            "Category": "Regulator",
        },
        {
            "Stakeholder Name": "Taiwanese Government",
            "Confidence Score": "92%",
            "Category": "Regulator",
        },
        {
            "Stakeholder Name": "hk elderly community",
            "Confidence Score": "88%",
            "Category": "Community",
        },
        {
            "Stakeholder Name": "hongkong elderly community",
            "Confidence Score": "90%",
            "Category": "Community",
        },
        {
            "Stakeholder Name": "Neighborhood Residents Group",
            "Confidence Score": "89%",
            "Category": "Community",
        },
        {
            "Stakeholder Name": "Local Residents Group",
            "Confidence Score": "89%",
            "Category": "Community",
        },
        {
            "Stakeholder Name": "UK Government",
            "Confidence Score": "95%",
            "Category": "Regulator",
        },
        {
            "Stakeholder Name": "USA Government",
            "Confidence Score": "92%",
            "Category": "Regulator",
        },
        {
            "Stakeholder Name": "united states of America Government",
            "Confidence Score": "92%",
            "Category": "Regulator",
        },
    ]

    merged = merge_stakeholders(stakeholders)

    print("✅ MERGED RESULTS:")
    for s in merged:
        print(f"\n{s['Canonical Name']} ({s['Cluster Size']})")
        print(f"  Category: {s['Category']} | Confidence: {s['Confidence Score']}")
        print(f"  Aliases: {s['Aliases']}")
        # print(f"  Match Score: {s['Match Score']}")


if __name__ == "__main__":
    main()


# ############################


# def normalize_match_key(name: str) -> str:
#     """normalization"""

#     govt_noise = (
#         r"\b(govt?|government|ministry|department|agency|authority|bureau|commission)\b"
#     )
#     name = re.sub(govt_noise, "", name, flags=re.I)

#     # Synonyms (preserve locations)
#     name = re.sub(r"\b(elderly|senior|old|aged)\b", "elderly", name, flags=re.I)
#     name = re.sub(
#         r"\b(community|association|group|network)\b", "community", name, flags=re.I
#     )

#     # Clean
#     name = re.sub(r"\s+", " ", name.lower().strip())
#     return name.strip()


# stakeholders = [
#     {
#         "Stakeholder Name": "Taiwan Government",
#         "Confidence Score": "95%",
#         "Category": "Regulator",
#     },
#     {
#         "Stakeholder Name": "Taiwanese Government",
#         "Confidence Score": "92%",
#         "Category": "Regulator",
#     },
#     {
#         "Stakeholder Name": "hk elderly community",
#         "Confidence Score": "88%",
#         "Category": "Community",
#     },
#     {
#         "Stakeholder Name": "hongkong elderly community",
#         "Confidence Score": "90%",
#         "Category": "Community",
#     },
#     {
#         "Stakeholder Name": "Neighborhood Residents Group",
#         "Confidence Score": "89%",
#         "Category": "Community",
#     },
#     {
#         "Stakeholder Name": "Local Residents Group",
#         "Confidence Score": "89%",
#         "Category": "Community",
#     },
#     # {
#     #     "Stakeholder Name": "uk Government",
#     #     "Confidence Score": "95%",
#     #     "Category": "Regulator",
#     # },
#     # {
#     #     "Stakeholder Name": "us Government",
#     #     "Confidence Score": "92%",
#     #     "Category": "Regulator",
#     # },
#     # {
#     #     "Stakeholder Name": "united states of America Government",
#     #     "Confidence Score": "92%",
#     #     "Category": "Regulator",
#     # },
# ]

# merged = []
# used = set()
# threshold = 80

# for i, s1 in enumerate(stakeholders):
#     print(f"i: {i}")
#     print(f"s1: {s1}")
#     print(f"used: {used}")
#     if i in used:
#         continue

#     s1_key = normalize_match_key(s1["Stakeholder Name"])
#     print(f"s1_key: {s1_key}")
#     cluster = [s1]
#     print(f"cluster: {cluster}")
#     for j, s2 in enumerate(stakeholders):
#         print(f"j: {j}, s2:{s2}")
#         if i == j or j in used:
#             continue

#         s2_key = normalize_match_key(s2["Stakeholder Name"])
#         print(f"s2_key: {s2_key}")
#         # Higher threshold for safety
#         print(f"SCORE:{fuzz.partial_ratio(s1_key, s2_key)}")
#         print(f"cluster: {cluster}")
#         print(f"used: {used}")
#         if fuzz.partial_ratio(s1_key, s2_key) >= threshold:
#             cluster.append(s2)
#             used.add(j)
#             print(f"cluster: {cluster}")
#             print(f"used: {used}")

#     # Merge logic (unchanged)
#     best = max(cluster, key=lambda x: float(x["Confidence Score"].strip("%")))
#     merged_record = best.copy()
#     merged_record["Canonical Name"] = best["Stakeholder Name"]
#     merged_record["Cluster Size"] = len(cluster)
#     merged_record["Aliases"] = [s["Stakeholder Name"] for s in cluster if s != best]
#     merged_record["Match Keys"] = [
#         normalize_match_key(s["Stakeholder Name"]) for s in cluster
#     ]

#     merged.append(merged_record)
#     used.add(i)

#     print(f"merged: {merged}")
#     print(f"used: {used}")

# # return merged
