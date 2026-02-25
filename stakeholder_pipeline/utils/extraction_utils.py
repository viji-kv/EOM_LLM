import json
import re
from typing import List, Dict
from zipfile import Path

# ===== IMPROVED JSON PARSER =====


def parse_json_response(raw_info: str) -> List[Dict]:
    """Robust JSON parser for LLM responses with multiple extraction strategies."""
    if not raw_info:
        print("        Empty raw response")
        return []

    # print(f"       Raw response preview: {repr(raw_info[:300])}...")

    # Strategy 1: Extract between ```json ... ```
    json_match = re.search(
        r"```json?\s*(\[.*?\])\s*```", raw_info, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        json_str = json_match.group(1)
        # print("       Strategy 1: Found ```json block")
    else:
        # Strategy 2: Extract largest JSON array candidate
        array_match = re.search(
            r"\[\s*(?:\{[^}]*\}|\d+|\[[^\]]*\]|\s*,\s*)*\s*\]", raw_info, re.DOTALL
        )
        if array_match:
            json_str = array_match.group(0)
            # print("       Strategy 2: Found array candidate")
        else:
            print("       No JSON block or array found")
            return []

    # Clean the extracted JSON
    json_str = re.sub(r"\\n", "", json_str)  # Remove escaped newlines
    json_str = re.sub(r"\\", "", json_str)  # Remove backslashes
    json_str = json_str.strip()

    # if len(json_str) < 10:
    #     print("        Extracted JSON too short")
    #     return []

    # print(f"       Cleaned JSON preview: {repr(json_str[:200])}...")

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            print(f"       Successfully parsed {len(parsed)} outputs")
            return parsed
        elif isinstance(parsed, dict):
            print("        Parsed single dict, wrapping in list")
            return [parsed]
        else:
            print(f"        Unexpected type: {type(parsed)}")
            return []
    except json.JSONDecodeError as e:
        print(f"       JSONDecodeError: {e}")
        print(
            f"       Error position preview: {repr(json_str[max(0, e.pos - 50) : e.pos + 50])}"
        )
        return []
    except Exception as e:
        print(f"       Unexpected error: {type(e).__name__}: {e}")
        return []


def calculate_splitter_params(model_context) -> tuple:
    # Use ~40-50% of context for chunk payload
    base_chunk = int(model_context * 0.50)
    CHARS_PER_TOKEN = 4  # English avg: 3.5-4.5 chars/token
    chunk_size = base_chunk * CHARS_PER_TOKEN

    # Overlap: 10-20% of chunk_size for semantic continuity
    overlap = int(chunk_size * 0.15)
    overlap = max(100, overlap)
    return chunk_size, overlap


# def save_output(result: Dict, output_filename: str, output_dir: str = "output") -> Path:
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     final_path = output_path / output_filename

#     with open(final_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     return final_path


# from pathlib import Path
# import re
# import json


# def save_output(result: Dict, output_filename: str, output_dir: str = "output") -> Path:
#     """
#     Saves results using absolute path resolution to avoid PermissionErrors.
#     """
#     # Force absolute path resolution
#     output_path = Path(output_dir).resolve()

#     # Create directory if it doesn't exist (exist_ok prevents errors)
#     output_path.mkdir(parents=True, exist_ok=True)

#     final_path = output_path / output_filename

#     # Use utf-8 and ensure_ascii=False for Chinese/Special characters
#     with open(final_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     return final_path


from pathlib import Path
from typing import Dict, Any
import json
import re


def save_output(result: Dict[str, Any], output_filename: str, output_dir) -> Path:
    """
    Save results to disk in a production-safe way.

    - Accepts output_dir as str or Path
    - Does not rely on the current working directory
    - Ensures directory exists
    """

    # Normalize output_dir to Path without forcing resolve()
    output_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Sanitize filename to be safe on Windows/macOS
    safe_filename = re.sub(r'[<>:"/\\|?*]', "_", output_filename)
    final_path = output_path / safe_filename

    # Write JSON (UTF-8, keep non-ASCII chars)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved: {final_path}")
    return final_path
