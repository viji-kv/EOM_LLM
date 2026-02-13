import asyncio
import openai  # Add if not present; assumes OPENAI_API_KEY in .env
from typing import List, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()


async def normalize_stakeholder_names(
    stakeholders: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    names = [s["Stakeholder Name"] for s in stakeholders if s["Stakeholder Name"]]
    # prompt = f"""
    # Normalize these stakeholder names into unique canonical forms. Merge variants (e.g., abbreviations, typos) based on likely matches.
    # Output ONLY a JSON array of objects: [{{"original": "list of variants", "canonical": "standard name"}}].
    # Names: {", ".join(names)}
    # """
    prompt = f"""
    Normalize these stakeholder names into unique canonical forms.

    Transliterate NON-ENGLISH stakeholder names to English Romanization first
    Then normalize variants into canonical English forms.
    
    Output EXACTLY this JSON: {{"normalized": [{{"original": ["var1", "var2"], "canonical": "Taiwan Government"}}, ...]}}

    Names: {", ".join(names)}
    """

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    # print("Raw LLM output:", content)
    parsed = json.loads(content)
    # print("Parsed type:", type(parsed), parsed)

    # Flexible: Try common keys or treat as direct list
    if isinstance(parsed, list):
        normalized = parsed
    elif "normalized" in parsed:
        normalized = parsed["normalized"]
    elif "canonical_names" in parsed or "result" in parsed:
        normalized = parsed.get("canonical_names") or parsed["result"]
    else:
        # Fallback: Assume direct dict/list from prompt
        normalized = parsed.get("normalized_list", parsed)

    # Map back: group originals by canonical
    canonical_map = {}
    for item in normalized:
        canon = item["canonical"]
        variants = item["original"]
        canonical_map[canon] = variants

    # Update stakeholders with canonical names (keep highest confidence per group)
    updated = []
    for s in stakeholders:
        for canon, variants in canonical_map.items():
            if s["Stakeholder Name"] in variants:
                s["Canonical Name"] = canon  # Add new field
                updated.append(s)
                break
    return updated


async def main():
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

    normalized = await normalize_stakeholder_names(stakeholders)
    print(normalized)
    print("\n\n\n")
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
