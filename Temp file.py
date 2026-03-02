            # NEW: Relationships array
            "relationships": {
                "type": "array",
                "description": "Direct relationships TO OTHER stakeholders mentioned in text",
                "items": {
                    "type": "object",
                    "properties": {
                        "related_stakeholder": {
                            "type": "string",
                            "description": "Exact name of related stakeholder (e.g., 'HKCSS')",
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "One word: regulates, funds, partners_with, competes_with, supplies_to, etc.",
                        },
                        "description": {
                            "type": "string",
                            "description": "How they interact (e.g., 'funds 70% of subvented homes')",
                        },
                        "confidence": {
                            "type": "string",
                            "description": "Confidence % (e.g., '90%')",
                        },
                    },
                    # "required": [
                    #     "related_stakeholder",
                    #     "relationship_type",
                    #     "confidence",
                    # ],
                },
            },
            # UPDATED: Pain points array
            "pain_points": {
                "type": "array",
                "description": "Structured pain points explicitly linked to stakeholders",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Short category (Funding, Staffing, Regulation, Capacity, etc.)",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the pain point",
                        },
                        "causing_stakeholders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Stakeholders that cause/contribute (optional)",
                        },
                        "stakeholder_impact": {
                            "type": "object",
                            "description": "How it affects each stakeholder",
                            "additionalProperties": {"type": "string"},
                            "minProperties": 1,
                        },
                        "confidence": {
                            "type": "string",
                            "description": "Confidence score (e.g., '92%')",
                        },
                    },
                    "required": [
                        "category",
                        "description",
                        "confidence",
                        "stakeholder_impact",
                    ],
                },
            },



###########################################


prompt=(
            "You are an expert assistant tasked with extracting stakeholders, their relationships, and structured pain points from the provided text using the extraction schema.\n\n"
            "Schema:\n{schema}\n\n"
            "Text:\n{topic}\n\n"
            "=== EXTRACTION INSTRUCTIONS ===\n"
            "1. STAKEHOLDERS: Extract ALL organizations, government bodies, NGOs, companies, and groups mentioned.\n"
            "   - Use EXACT names from text (e.g., 'Social Welfare Department (SWD)' → Canonical: 'Social Welfare Department')\n"
            "   - Categories: Regulator, Partner, Supplier, Consumer, Competitor, Provider, Government, NGO, Private\n"
            "   - Role: Specific function from text (max 100 chars)\n\n"
            "2. RELATIONSHIPS: For each stakeholder, extract DIRECT relationships to OTHER stakeholders:\n"
            "   - related_stakeholder: Exact name from text\n"
            "   - relationship_type: Single word/action (funds, regulates, partners_with, competes_with, supplies_to)\n"
            "   - Examples: 'SWD funds HKCSS' → {'related_stakeholder': 'HKCSS', 'relationship_type': 'funds'}\n"
            "   - 'HA competes with private hospitals' → {'related_stakeholder': 'Private Residential Care Homes', 'relationship_type': 'competes_with'}\n\n"
            "3. PAIN POINTS (CRITICALLY IMPORTANT): Identify ALL challenges mentioned and link to SPECIFIC stakeholders:\n"
            "   - category: Funding, Staffing, Capacity, Regulation, Access, Quality, Demand\n"
            "   - description: Brief problem statement\n"
            "   - causing_stakeholders: Who causes it (optional, only if explicit)\n"
            '   - stakeholder_impact: {"Stakeholder Name": "specific impact description"}\n'
            "   - Examples:\n"
            '     {"category": "Staffing", "description": "NGOs report 20% vacancies", '
            '      "causing_stakeholders": ["SWD"], "stakeholder_impact": {"Baptist Oi Kwan": "Service demand unmet"}, "confidence": "93%"}\n\n'
            "4. SOURCE METADATA:\n"
            "   - extraction_context: Copy 200-300 chars AROUND each stakeholder mention\n"
            "   - extraction_summary: 2-3 sentence doc summary + key stakeholder interactions\n\n"
            "=== CRITICAL RULES ===\n"
            "• Extract ALL mentions, even brief ones\n"
            "• Use ONLY information explicitly in text - NO assumptions/external knowledge\n"
            "• Relationships: Only between stakeholders BOTH mentioned in this chunk\n"
            "• Pain points: Must reference specific stakeholders by name\n"
            "• Confidence: 85-95% for clear mentions, 70-84% for implied\n"
            "• Output VALID JSON array matching schema exactly\n\n"
            "Provide answer as clean, parseable JSON."
        ),


chunks = ["hello", "hi", "how are you", "what is your name"]
for i, chunk in enumerate(chunks):
        print(i, chunk)

Text = "Hi hello , how are you"

print(f"Contains 'Hi': {'you' in Text}")

pain_point = {}
pain_point.setdefault('affected_stakeholders', [])
pain_point.setdefault('severity', 'Medium')