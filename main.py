import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from typer import prompt

def load_env_file():
    """Auto-loads variables from .env file in the same directory."""
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v

load_env_file()

app = FastAPI(
    title="SatQuery AI Backend",
    description="Interactive AI Assistant for Satellite Image Analysis (Knowledge-Base Grounded)",
    version="2.0.0",
)

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    answer: str
    mode: str
    scenario: Optional[str] = None
    image_id: Optional[str] = None


# ============================================================================
# 1. Structured Knowledge Base & Image Recognition Engine
# ============================================================================

def compute_image_dhash(image_bytes: bytes, hash_size: int = 8) -> str:
    """Computes a 64-bit difference hash (dHash) for visual image comparison."""
    try:
        image = (
            Image.open(io.BytesIO(image_bytes))
            .convert("L")
            .resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        )
        pixels = list(image.getdata())
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)

        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
                decimal_value = 0
        return "".join(hex_string)
    except Exception as e:
        print(f"Warning: Could not compute dHash: {e}")
        return ""


def hamming_distance(hex1: str, hex2: str) -> int:
    """Calculates bitwise Hamming distance between two 64-bit hex hash strings."""
    if not hex1 or not hex2 or len(hex1) != len(hex2):
        return 999
    try:
        b1 = bin(int(hex1, 16))[2:].zfill(64)
        b2 = bin(int(hex2, 16))[2:].zfill(64)
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))
    except Exception:
        return 999


import re as _re

def _normalize_filename(name: str) -> str:
    """Normalize a filename for fuzzy matching.

    Rules (order matters):
    1. Lowercase + strip whitespace
    2. Strip the extension, normalize the stem, reattach the extension
    3. Remove duplicate-upload suffixes: (1), (2), (3) …
    4. Treat underscores and hyphens as spaces
    5. Collapse multiple spaces to one
    """
    name = name.lower().strip()
    # Split off extension so we don't mangle it
    dot_pos = name.rfind(".")
    if dot_pos != -1:
        stem, ext = name[:dot_pos], name[dot_pos:]   # ext includes the dot
    else:
        stem, ext = name, ""

    # Remove (1), (2) … duplicate-upload markers (with optional surrounding space)
    stem = _re.sub(r"\s*\(\d+\)\s*", "", stem)
    # Treat _ and - as spaces
    stem = stem.replace("_", " ").replace("-", " ")
    # Collapse multiple spaces
    stem = _re.sub(r"\s+", " ", stem).strip()

    return stem + ext


class KnowledgeBaseManager:
    """Loads JSON scenario files and matches uploaded images to known knowledge."""

    def __init__(self, kb_dir: str = "knowledge_base", img_dir: str = "images"):
        self.kb_dir = Path(kb_dir)
        self.img_dir = Path(img_dir)
        self.scenarios: Dict[str, dict] = {}
        self.reference_hashes: Dict[str, str] = {}
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Loads all JSON knowledge base files into memory and indexes reference images."""
        if not self.kb_dir.exists():
            print(f"Warning: Knowledge base directory '{self.kb_dir}' does not exist.")
            return

        for json_path in self.kb_dir.glob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    image_id = data.get("image_id", json_path.stem)
                    self.scenarios[image_id] = data
                    print(f"Loaded scenario: {data.get('scenario')} ({image_id})")
            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        # Index reference image hashes
        image_search_paths = [self.img_dir, Path(".")]
        for img_id, data in self.scenarios.items():
            possible_filenames = [data.get("filename", "")] + data.get("aliases", [])
            for p_name in possible_filenames:
                if not p_name:
                    continue
                for search_dir in image_search_paths:
                    candidate_file = search_dir / p_name
                    if candidate_file.exists() and candidate_file.is_file():
                        try:
                            with open(candidate_file, "rb") as f:
                                h = compute_image_dhash(f.read())
                                if h and img_id not in self.reference_hashes:
                                    self.reference_hashes[img_id] = h
                                    print(f"Indexed visual fingerprint for [{img_id}] from {candidate_file}")
                        except Exception as e:
                            print(f"Could not hash {candidate_file}: {e}")

    def identify_image(
        self, image_bytes: bytes, filename: Optional[str] = None
    ) -> Tuple[Optional[dict], str, int]:
        """
        Multi-stage image identification:
          Stage 1: Normalized exact filename match      → confidence 100
          Stage 2: Normalized exact alias match         → confidence 95
          Stage 3: Visual perceptual hash (dHash)       → distance-based
          Stage 4: Flexible substring / keyword match   → confidence 85
        Returns (matched_data, identification_method, confidence_or_distance)
        """

        # ── Stage 1 & 2: Filename / alias matching (normalized) ─────────────
        if filename:
            cleaned_name = _normalize_filename(filename)

            for data in self.scenarios.values():
                # Stage 1 – exact normalized filename
                target_fn = _normalize_filename(data.get("filename", ""))
                if target_fn and cleaned_name == target_fn:
                    return data, "filename_match", 100

                # Stage 2 – exact normalized alias
                for alias in data.get("aliases", []):
                    if _normalize_filename(alias) == cleaned_name:
                        return data, "alias_match", 95

        # ── Stage 3: Visual perceptual hash (dHash) ─────────────────────────
        if image_bytes and self.reference_hashes:
            query_hash = compute_image_dhash(image_bytes)
            if query_hash:
                best_match_id = None
                lowest_distance = 999

                for ref_id, ref_hash in self.reference_hashes.items():
                    dist = hamming_distance(query_hash, ref_hash)
                    if dist < lowest_distance:
                        lowest_distance = dist
                        best_match_id = ref_id

                # <= 15 Hamming distance → high visual similarity
                if lowest_distance <= 15 and best_match_id:
                    return self.scenarios[best_match_id], "visual_hash_match", lowest_distance

        # ── Stage 4: Flexible substring / keyword match ──────────────────────
        if filename:
            fn_norm = _normalize_filename(filename)
            for data in self.scenarios.values():
                for alias in data.get("aliases", []):
                    alias_norm = _normalize_filename(alias)
                    if alias_norm and (alias_norm in fn_norm or fn_norm in alias_norm):
                        return data, "keyword_match", 85

        return None, "unmatched", 0


# Initialize knowledge base manager
kb_manager = KnowledgeBaseManager()


# ============================================================================
# 2. Grounded Vision AI Prompt Builder
# ============================================================================
def build_grounded_prompt(kb_data: Optional[dict], question: str) -> str:
    """Build a strict vision-first prompt for satellite and aerial image analysis."""

    context_section = ""

    if kb_data:
        scenario = kb_data.get("scenario", "Satellite Observation")
        location = kb_data.get("location_context", "Target Region")
        description = kb_data.get("detailed_description", "")

        context_section = f"""
REFERENCE METADATA — USE ONLY AS SECONDARY CONTEXT

Scenario: {scenario}
Geographic Setting: {location}
Reference Summary: {description}

IMPORTANT:
Reference metadata must NEVER override what is visibly present in the
uploaded image. If the metadata conflicts with the image, trust the image.
"""

    prompt = f"""
You are Orbital AI, an Earth Observation and Satellite Image Analysis Assistant.

Your primary task is to analyze ONLY what is visually observable in the
uploaded image.

{context_section}

USER QUESTION:
"{question}"

STRICT ANALYSIS RULES:

1. Examine the actual uploaded image before answering.

2. Describe only features that are visually supported by the image.

3. NEVER invent:
   - percentages
   - exact measurements
   - NDVI values
   - vegetation coverage values
   - population or urban density
   - locations
   - roads, buildings, parks, rivers, lakes, or other objects that are not clearly visible.

4. Do NOT assume the image belongs to an urban, agricultural, forest,
   flooding, or other predefined scenario unless the visual evidence supports it.

5. If something cannot be determined reliably from the image, explicitly say:
   "This cannot be determined reliably from the visible image."

6. If the image is unclear, heavily cropped, obscured, a screenshot,
   or does not contain enough visual information, say so instead of guessing.

7. For questions about vegetation:
   distinguish between clearly visible vegetation and exposed soil,
   sand, rock, barren terrain, clouds, or image artifacts.

8. For questions about water:
   identify water only when there is visible evidence consistent with a
   water body. Do not confuse shadows, dark terrain, clouds, or image artifacts
   with water.

9. Do not provide numerical estimates unless the image or supplied metadata
   explicitly provides the numbers.

10. Keep the answer concise, factual, and directly focused on the user's question.

Return the response in this format:

### 🛰️ Visual Analysis

- **Observation:** ...
- **Evidence:** ...
- **Confidence:** High / Medium / Low

### 📌 Answer

A direct answer to the user's question.
"""

    return prompt.strip()


# ============================================================================
# 3. Intelligent Local Fallback Engine (Demo / Offline Mode)
# ============================================================================

def generate_local_knowledge_response(kb_data: dict, question: str) -> str:
    """Directly answers user queries from structured JSON data when no API key is set."""
    q = question.lower()
    scenario = kb_data.get("scenario", "Satellite Observation")
    location = kb_data.get("location_context", "Monitored Region")
    
    # 1. Water / Flooding / Inundation queries
    if any(k in q for k in ["water", "flood", "inundat", "submerg", "river", "lake", "ocean", "sea"]):
        if "flood" in scenario.lower():
            impacts = kb_data.get("possible_environmental_impacts", [])
            obs = kb_data.get("predefined_observations", {})
            return f"""### 🌊 Flood & Water Inundation Analysis: {scenario}

* **Inundation Extent**: {kb_data.get('land_use_information', {}).get('water_body_coverage', 'Extensive standing floodwaters')}
* **Water Depth & Severity**: {obs.get('estimated_inundation_depth', '0.6m to 2.4m')} ({obs.get('flood_severity_index', 'Critical')})
* **Submerged Land**: {obs.get('affected_land_area_estimate', 'Over 340 hectares inundated')}
* **Key Observations**:
{chr(10).join(f'  * {f}' for f in kb_data.get('visible_features', [])[:3])}
* **Environmental Impact**:
{chr(10).join(f'  * {imp}' for imp in impacts[:2])}"""
        else:
            land_use = kb_data.get("land_use_information", {})
            water_info = land_use.get("water_body_coverage") or land_use.get("rural_roads_and_canals", "No catastrophic flooding detected.")
            return f"""### 🌊 Hydrological Analysis: {scenario}

* **Water Feature Assessment**: {water_info}
* **Setting**: {location}
* **Context**: The imagery shows localized hydrological features consistent with normal scenario conditions."""

    # 2. Deforestation / Forest Loss / Vegetation queries
    if any(k in q for k in ["forest", "deforest", "tree", "vegetation", "canopy", "green", "crop", "ndvi"]):
        if "deforest" in scenario.lower():
            obs = kb_data.get("predefined_observations", {})
            return f"""### 🌲 Forest Canopy & Deforestation Assessment: {scenario}

* **Primary Canopy Loss**: {obs.get('estimated_cleared_area', '185 hectares cleared in active sector')}
* **Driver of Loss**: {obs.get('primary_driver', 'Commercial expansion and road clearing')}
* **Canopy Health (NDVI)**: Healthy canopy NDVI: {obs.get('average_ndvi_healthy_canopy', '0.80')} vs. Cleared plots: {obs.get('average_ndvi_cleared_patches', '0.15')}
* **Pattern Identified**:
{chr(10).join(f'  * {c}' for c in kb_data.get('detected_changes_and_patterns', [])[:2])}
* **Ecological Impact**:
{chr(10).join(f'  * {imp}' for imp in kb_data.get('possible_environmental_impacts', [])[:2])}"""
        elif "agri" in scenario.lower():
            obs = kb_data.get("predefined_observations", {})
            return f"""### 🌾 Agricultural Vegetation & Crop Health: {scenario}

* **Crop Health Status**: {obs.get('crop_health_status', 'Generally optimal')}
* **Dominant Crops**: {obs.get('dominant_crop_types', 'Alfalfa, corn, grain rotations')}
* **Cultivated Area**: {kb_data.get('land_use_information', {}).get('active_cultivated_cropland', '68.4%')}
* **Identified Patterns**:
{chr(10).join(f'  * {f}' for f in kb_data.get('visible_features', [])[:3])}"""
        else:
            return f"""### 🌿 Vegetation Cover Analysis: {scenario}

* **Location Context**: {location}
* **Vegetation Details**:
{chr(10).join(f'  * {f}' for f in kb_data.get('visible_features', []) if 'vegetation' in f.lower() or 'park' in f.lower() or 'green' in f.lower()) or '  * Scene contains structured green cover as indicated in land-use breakdown.'}"""

    # 3. Urban Development / Buildings / Infrastructure queries
    if any(k in q for k in ["urban", "rural", "building", "road", "infrastructure", "city", "street", "construction"]):
        if "urban" in scenario.lower():
            obs = kb_data.get("predefined_observations", {})
            land_use = kb_data.get("land_use_information", {})
            return f"""### 🏙️ Urban Development & Infrastructure Analysis: {scenario}

* **Urban Density**: {obs.get('urbanization_index', 'Very High')} ({land_use.get('impervious_built_up_structures', '52.4% built-up')})
* **Road Density**: {obs.get('road_network_density', '14.2 km/km²')}
* **Visible Infrastructure**:
{chr(10).join(f'  * {f}' for f in kb_data.get('visible_features', [])[:3])}
* **Development Patterns**:
{chr(10).join(f'  * {c}' for c in kb_data.get('detected_changes_and_patterns', [])[:2])}"""
        else:
            return f"""### 🏗️ Infrastructure & Land Classification: {scenario}

* **Classification**: {location}
* **Land Use Profile**:
{chr(10).join(f'  * {k.replace("_", " ").title()}: {v}' for k, v in kb_data.get('land_use_information', {}).items())}
* **Transport / Roads**:
{chr(10).join(f'  * {f}' for f in kb_data.get('visible_features', []) if 'road' in f.lower() or 'track' in f.lower() or 'canal' in f.lower()) or '  * Rural transport access observed.'}"""

    # 4. Environmental Impact / Recommendations queries
    if any(k in q for k in ["impact", "environment", "risk", "hazard", "action", "recommend", "damage"]):
        obs = kb_data.get("predefined_observations", {})
        return f"""### ⚠️ Environmental Impacts & Action Plan: {scenario}

* **Potential Environmental Impacts**:
{chr(10).join(f'  * {imp}' for imp in kb_data.get('possible_environmental_impacts', []))}
* **Recommended Actions**:
  * {obs.get('recommended_actions', 'Continue satellite tracking and field validation.')}"""

    # 5. General Overview / Describe queries
    if any(k in q for k in ["describe", "overview", "what", "tell", "summary", "see", "show", "analyze"]):
        obs = kb_data.get("predefined_observations", {})
        return f"""### 🛰️ Satellite Scene Overview: {scenario}

* **Location Context**: {location}
* **Scene Summary**: {kb_data.get('detailed_description', '')}

#### Key Visible Features:
{chr(10).join(f'* {f}' for f in kb_data.get('visible_features', []))}

#### Land-Use Breakdown:
{chr(10).join(f'* **{k.replace("_", " ").title()}**: {v}' for k, v in kb_data.get('land_use_information', {}).items())}

#### Predefined Observations:
{chr(10).join(f'* **{k.replace("_", " ").title()}**: {v}' for k, v in obs.items())}"""

    # 6. Unanswerable / Out of scope queries
    return f"""### 🛰️ SatQuery AI Knowledge Response

The available satellite image knowledge for **{scenario}** does not provide sufficient information to answer your specific question.

The calibrated knowledge base for this scene includes:
* **Scenario & Setting**: {scenario} ({location})
* **Visual Features**: {len(kb_data.get('visible_features', []))} verified terrain and structural elements
* **Land-Use Metrics**: Measured coverage ratios for water, vegetation, and built structures
* **Environmental Impacts & Recommendations**: Verified post-capture observations

*Try asking about visible land-use, buildings, roads, vegetation cover, environmental impacts, or scene descriptions.*"""


def generate_uncalibrated_demo_response(
    filename: str,
    question: str,
    image_bytes: bytes,
) -> str:
    """Returns a useful offline response for uploads that do not match calibrated scenarios."""
    q = question.lower()
    image_summary = "The image was received successfully, but it does not match a calibrated demo scenario."
    image_kind_note = "Offline demo mode cannot verify exact objects in arbitrary custom uploads."

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            image_summary = f"The image was received successfully at {width} x {height} px."
    except Exception:
        pass

    if "screenshot" in filename.lower():
        image_kind_note = (
            "This upload appears to be a screenshot of the app or browser, not raw satellite "
            "or aerial imagery. Demo mode is responding correctly, but it cannot perform "
            "satellite land-cover analysis on a UI screenshot."
        )

    if any(k in q for k in ["water", "flood", "river", "lake", "ocean", "sea"]):
        focus = "water bodies, flooding, river channels, shoreline boundaries, and wet or submerged areas"
    elif any(k in q for k in ["forest", "deforest", "tree", "vegetation", "canopy", "green", "crop", "ndvi"]):
        focus = "vegetation cover, canopy breaks, crop patterns, exposed soil, and land-clearing signals"
    elif any(k in q for k in ["urban", "building", "road", "infrastructure", "city", "street", "construction"]):
        focus = "roads, building clusters, impervious surfaces, construction zones, and access routes"
    elif any(k in q for k in ["impact", "environment", "risk", "hazard", "action", "recommend", "damage"]):
        focus = "environmental risk, visible disturbance, likely impacts, and follow-up monitoring actions"
    else:
        focus = "major land-cover classes, visible structures, vegetation, water, and disturbed surfaces"

    title = (
        "Screenshot Upload Detected"
        if "screenshot" in filename.lower()
        else "Demo Mode Response: Custom Upload"
    )

    return f"""### {title}

* **Upload Status**: {image_summary}
* **Current Limitation**: {image_kind_note}
* **Question Focus**: I would inspect **{focus}** for this request.
* **Best Demo Path**: Click one of the calibrated buttons: **Flooding**, **Deforestation**, **Urban Grid**, or **Agriculture**, then press **Analyze Image** for a full offline prototype answer.
* **For Real Custom Images**: Add a working Gemini API key to enable live vision analysis on any uploaded satellite image."""


# ============================================================================
# 4. Gemini AI Model Caller (Grounded Multimodal)
# ============================================================================

def generate_gemini_analysis(
    image_bytes: bytes,
    mime_type: Optional[str],
    kb_data: Optional[dict],
    question: str,
) -> str:

    from google import genai
    from google.genai import types

    load_env_file()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    prompt = build_grounded_prompt(kb_data, question)
    print("\n========== GEMINI PROMPT ==========")
    print(prompt)
    print("===================================\n")

    candidate_models = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]

    last_exception = None

    for model_name in candidate_models:
        try:

            print(f"[Gemini AI] Running model: {model_name}")
            print(f"[Gemini AI] MIME type: {mime_type}")
            print(f"[Gemini AI] Image size: {len(image_bytes)} bytes")

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=(
                            mime_type
                            if mime_type and mime_type.startswith("image/")
                            else "image/png"
                        ),
                    ),
                    prompt,
                ],
            )

            if response and response.text:
                print(
                    f"[Gemini AI] Successfully generated response "
                    f"using model '{model_name}'."
                )

                return response.text

            print("[Gemini Warning] Gemini returned an empty response.")

        except Exception as err:

            last_exception = err

            print(
                f"[Gemini Warning] Model '{model_name}' failed: {err}"
            )

    if last_exception:
        raise last_exception

    return "No analysis text could be generated by the model."
# ============================================================================
# 5. API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "SatQuery AI",
        "version": "2.0.0",
        "loaded_scenarios": list(kb_manager.scenarios.keys()),
        "scenario_count": len(kb_manager.scenarios),
    }


@app.get("/scenarios")
def list_scenarios():
    """Returns summary list of all calibrated knowledge base scenarios."""
    return [
        {
            "image_id": data.get("image_id"),
            "scenario": data.get("scenario"),
            "filename": data.get("filename"),
            "location_context": data.get("location_context"),
        }
        for data in kb_manager.scenarios.values()
    ]


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    if not image:
        raise HTTPException(
            status_code=400,
            detail="Image file is required.",
        )

    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    image_bytes = await image.read()
    orig_filename = image.filename or ""

    # 1. Identify which calibrated scenario (if any) this image matches
    kb_data, match_method, confidence = kb_manager.identify_image(
        image_bytes=image_bytes, filename=orig_filename
    )

    print(
        f"Image Uploaded: '{orig_filename}', Match: {kb_data.get('image_id') if kb_data else 'None'}, "
        f"Method: {match_method}, Metric: {confidence}"
    )

    scenario_name = kb_data.get("scenario", "Satellite Observation") if kb_data else "Custom Satellite Capture"
    image_id = kb_data.get("image_id") if kb_data else None

    load_env_file()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # 2. If Gemini API key is available, run direct Vision AI inference on ANY image
    if gemini_api_key:
        try:
            print(f"[SatQuery AI] Running Gemini Vision Analysis on [{orig_filename}]...")
            answer = generate_gemini_analysis(
                image_bytes=image_bytes,
                mime_type=image.content_type,
                kb_data=kb_data,
                question=question,
            )
            return AnalysisResponse(
                answer=answer,
                mode="Gemini AI (Vision-Grounded)",
                scenario=scenario_name,
                image_id=image_id,
            )
        except Exception as e:
            print(f"[SatQuery Error] Gemini API failed with error: {e}")
            print("[SatQuery AI] Falling back to local knowledge engine.")
    else:
        print("[SatQuery AI] GEMINI_API_KEY not found in environment or .env file. Running in Demo Mode.")

    # 3. Fallback to local Knowledge-Base engine (Demo Mode)
    if kb_data:
        answer = generate_local_knowledge_response(kb_data=kb_data, question=question)
        return AnalysisResponse(
            answer=answer,
            mode="SatQuery Knowledge Engine (Demo Mode)",
            scenario=scenario_name,
            image_id=image_id,
        )
    else:
        unrecognized_msg = generate_uncalibrated_demo_response(
            filename=orig_filename,
            question=question,
            image_bytes=image_bytes,
        )
        return AnalysisResponse(
            answer=unrecognized_msg,
            mode="SatQuery Knowledge Engine (Demo Mode)",
            scenario="Custom Upload",
            image_id=None,
        )


if __name__ == "__main__":
    import uvicorn

    load_env_file()
    key = os.getenv("GEMINI_API_KEY")

    print("\n" + "=" * 60)
    print("🛰️   SatQuery AI Backend Server")
    print("🌐   API URL: http://127.0.0.1:8000")
    if key and key.strip():
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        print(f"✅   Gemini AI: ENABLED (Key loaded: {masked_key})")
    else:
        print("⚠️   Gemini AI: DEMO MODE (GEMINI_API_KEY is not set)")
        print("💡   To enable Gemini, paste your key into the .env file in this folder:")
        print("     GEMINI_API_KEY=AIzaSy...")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
