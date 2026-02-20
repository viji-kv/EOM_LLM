import json


input_file = (
    "output/stakeholders_output_AP HK - Financial industry_llm_only_consolidated.json"
)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
