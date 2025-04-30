import json
import re
import logging
from typing import Dict, Any, List, Optional
# from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError, validator
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai
from dotenv import load_dotenv
from functools import partial
# For converting Pydantic models in the final output if needed
from pydantic.json import pydantic_encoder
import os
from datetime import datetime

load_dotenv()
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')

try:
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    logging.error(f"Failed to configure Gemini: {e}.")
    exit(1)

MODEL_NAME = 'gemini-1.5-flash-latest'


class Segment(BaseModel):
    speaker: str
    text: str

class SegmentationResponse(BaseModel):
    segments: List[Segment]

class ProofreadQualityResponse(BaseModel):
    proofread_text: str # Expecting the text in the "[speaker]: text" format
    quality_score: float

    @validator('quality_score')
    def score_must_be_in_range(cls, v):
        # Allow None if quality check fails, but validate if present
        if v is not None and not (1 <= v <= 10):
            raise ValueError(f'Quality score {v} must be between 1 and 10')
        return v

class AppState(BaseModel):
    input_text: str
    segments: Optional[List[Segment]] = None
    proofread_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


def supervisor_node(state: AppState) -> Dict[str, Any]:
    """
    Supervises the workflow and performs initial validation.
    """
    logging.info("Supervisor: Starting validation.")
    if not state.input_text or not state.input_text.strip():
        logging.error("Supervisor: Input text is empty.")
        return {"error": "Input text cannot be empty"}

    char_count = len(state.input_text)
    word_count = len(state.input_text.split())

    logging.info(f"Supervisor: Input validated. Chars: {char_count}, Words: {word_count}")
    updated_metadata = state.metadata.copy()
    updated_metadata.update({
        "initial_char_count": char_count,
        "initial_word_count": word_count,
        "validation_passed": True
    })
    return {"metadata": updated_metadata, "error": None}

def segment_agent(state: AppState, model: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Segments the input text using an LLM.
    """
    logging.info("Segmenter: Starting text segmentation.")
    if state.error:
         logging.warning("Segmenter: Skipping due to previous error.")
         return {}

    segment_prompt = f"""
    Analyze the following text and segment it based on who is speaking or if it's narration.
    Identify distinct speakers (use consistent names like "Alice", "Bob", "Narrator").

    Text to segment:
    "{state.input_text}"

    Return the output STRICTLY in this JSON format:
    {{
        "segments": [
            {{
                "speaker": "speaker_name_or_Narrator",
                "text": "The corresponding text segment."
            }},
            ...
        ]
    }}
    Return ONLY the raw JSON object, without any markdown formatting (like ```json ... ```) or introductory text.
    """
    try:
        response = model.generate_content(segment_prompt)
        response_text = response.text.strip()

        # Clean potential markdown fences
        if response_text.startswith("```json"):
            response_text = re.sub(r"^```json\s*", "", response_text, flags=re.IGNORECASE)
            response_text = re.sub(r"\s*```$", "", response_text)
        elif response_text.startswith("```"):
             response_text = re.sub(r"^```\s*", "", response_text)
             response_text = re.sub(r"\s*```$", "", response_text)

        logging.debug(f"Segmenter: Raw LLM response:\n{response_text}")

        try:
            raw_parsed = json.loads(response_text)
            validated_response = SegmentationResponse.parse_obj(raw_parsed)
            segments = validated_response.segments

            logging.info(f"Segmenter: Text segmented successfully into {len(segments)} chunks.")
            updated_metadata = state.metadata.copy()
            updated_metadata.update({
                "num_segments": len(segments),
                "segmentation_successful": True,
                "segmentation_error": None
            })
            # Store Pydantic objects directly in state
            return {
                "segments": segments,
                "metadata": updated_metadata,
                "error": None
            }

        except json.JSONDecodeError as e:
            error_msg = f"JSON Decode Error: {e}. Response: '{response_text[:200]}...'"
            logging.error(f"Segmenter: {error_msg}")
            updated_metadata = state.metadata.copy()
            updated_metadata.update({"segmentation_successful": False, "segmentation_error": error_msg})
            return {"metadata": updated_metadata, "error": "Failed to decode segmentation JSON"}

        except ValidationError as e:
             error_msg = f"JSON Structure Error: {e}. Parsed: '{raw_parsed}'"
             logging.error(f"Segmenter: {error_msg}")
             updated_metadata = state.metadata.copy()
             updated_metadata.update({"segmentation_successful": False, "segmentation_error": error_msg})
             return {"metadata": updated_metadata, "error": "Invalid segmentation JSON structure"}

    except Exception as e:
        error_msg = f"Unexpected segmentation error: {e}"
        logging.exception("Segmenter: An unexpected error occurred.")
        updated_metadata = state.metadata.copy()
        updated_metadata.update({"segmentation_successful": False, "segmentation_error": error_msg})
        return {"metadata": updated_metadata, "error": "Unexpected error during segmentation"}


def combined_proofread_quality_agent(state: AppState, model: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Proofsreads the segmented text and assesses its quality in a single LLM call.
    """
    logging.info("Proofreader/Quality: Starting combined step.")
    if state.error:
         logging.warning("Proofreader/Quality: Skipping due to previous error.")
         return {}
    if not state.segments:
         logging.warning("Proofreader/Quality: Skipping - No segments available.")
         updated_metadata = state.metadata.copy()
         updated_metadata.update({"proofreading_skipped": True, "quality_check_skipped": True})
         return {"proofread_text": "", "metadata": updated_metadata}

    # Format text with speaker context for the LLM
    formatted_text_for_llm = "\n".join([
        f"[{segment.speaker}]: {segment.text}" for segment in state.segments
    ])
    logging.debug(f"Proofreader/Quality: Sending formatted text:\n{formatted_text_for_llm}")

    combined_prompt = f"""
# Task 1: Proofread Segmented Text
Review the following text, which is segmented by speaker/narrator using "[speaker]: text" format.
Correct grammar, spelling, punctuation, clarity, and flow within each segment's text.
Maintain the original speaker attribution tags exactly (e.g., "[Narrator]:", "[Alice]:"). DO NOT change them.

Text to process:
{formatted_text_for_llm}

# Task 2: Assess Quality
After proofreading, assess the *overall quality* of the corrected text (grammar, clarity, coherence).
Provide a quality score from 1 (very poor) to 10 (excellent), allowing decimals (e.g., 8.5).

# Output Format
Return your response STRICTLY as a single JSON object with two keys:
1. "proofread_text": A string containing the entire corrected text, preserving the "[speaker]: text" format for each line. Use newline characters (\\n) between segments.
2. "quality_score": A float representing the numerical quality score (1-10).

Example Output Format:
{{
  "proofread_text": "[Narrator]: The wind howled. It was cold.\\n[Alice]: I think it's time to go inside!",
  "quality_score": 8.5
}}
Return ONLY the raw JSON object. No explanations, preamble, or markdown formatting.
"""
    try:
        response = model.generate_content(combined_prompt)
        response_text = response.text.strip()

        # Clean potential markdown fences
        if response_text.startswith("```json"):
            response_text = re.sub(r"^```json\s*", "", response_text, flags=re.IGNORECASE)
            response_text = re.sub(r"\s*```$", "", response_text)
        elif response_text.startswith("```"):
             response_text = re.sub(r"^```\s*", "", response_text)
             response_text = re.sub(r"\s*```$", "", response_text)

        logging.debug(f"Proofreader/Quality: Raw LLM response:\n{response_text}")

        try:
            raw_parsed = json.loads(response_text)
            # Use .parse_obj for dictionary parsing, .parse_raw for raw JSON string
            validated_response = ProofreadQualityResponse.parse_obj(raw_parsed)

            proofread_segmented_text = validated_response.proofread_text
            quality_score = validated_response.quality_score

            # Post-process the proofread text to get a clean combined version
            clean_proofread_lines = []
            for line in proofread_segmented_text.splitlines():
                if line.strip():
                    match = re.match(r"\[(.*?)\]:\s*(.*)", line)
                    if match:
                        speaker, text = match.groups()
                        clean_proofread_lines.append(text.strip())
                    else:
                        logging.warning(f"Proofreader/Quality: Line didn't match expected format: '{line}'. Appending as is.")
                        clean_proofread_lines.append(line.strip())

            final_proofread_text = " ".join(clean_proofread_lines)

            logging.info(f"Proofreader/Quality: Processing successful. Quality Score: {quality_score}")
            updated_metadata = state.metadata.copy()
            updated_metadata.update({
                "proofreading_successful": True,
                "quality_check_successful": True,
                "quality_score": quality_score, # Store the score here
                "proofread_length": len(final_proofread_text),
                "combined_step_error": None
            })

            return {
                "proofread_text": final_proofread_text, # Return combined text
                "metadata": updated_metadata, # Metadata contains the score
                "error": None
            }

        except json.JSONDecodeError as e:
            error_msg = f"Combined JSON Decode Error: {e}. Response: '{response_text[:200]}...'"
            logging.error(f"Proofreader/Quality: {error_msg}")
            updated_metadata = state.metadata.copy()
            updated_metadata.update({"proofreading_successful": False, "quality_check_successful": False, "combined_step_error": error_msg, "quality_score": None})
            return {"metadata": updated_metadata, "error": "Failed to decode combined proofread/quality JSON"}

        except ValidationError as e:
             error_msg = f"Combined JSON Structure/Value Error: {e}. Parsed: '{raw_parsed}'"
             logging.error(f"Proofreader/Quality: {error_msg}")
             updated_metadata = state.metadata.copy()
             updated_metadata.update({"proofreading_successful": False, "quality_check_successful": False, "combined_step_error": error_msg, "quality_score": None})
             return {"metadata": updated_metadata, "error": "Invalid combined proofread/quality JSON structure"}

    except Exception as e:
        error_msg = f"Unexpected combined step error: {e}"
        logging.exception("Proofreader/Quality: An unexpected error occurred.")
        updated_metadata = state.metadata.copy()
        updated_metadata.update({"proofreading_successful": False, "quality_check_successful": False, "combined_step_error": error_msg, "quality_score": None})
        return {"metadata": updated_metadata, "error": "Unexpected error during proofread/quality step"}


def create_workflow() -> StateGraph:
    """
    Creates and configures the workflow graph.
    """
    logging.info("Creating workflow graph...")
    try:
        gemini_model = genai.GenerativeModel(MODEL_NAME)
        logging.info(f"Gemini model '{MODEL_NAME}' instantiated successfully.")
    except Exception as e:
        logging.error(f"Failed to instantiate Gemini model '{MODEL_NAME}': {e}")
        raise

    graph = StateGraph(AppState)

    bound_segment_agent = partial(segment_agent, model=gemini_model)
    bound_combined_agent = partial(combined_proofread_quality_agent, model=gemini_model)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("segmenter", bound_segment_agent)
    graph.add_node("proofreader_quality", bound_combined_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "segmenter")
    graph.add_edge("segmenter", "proofreader_quality")
    graph.add_edge("proofreader_quality", END)

    workflow = graph.compile()
    logging.info("Workflow graph compiled successfully.")
    return workflow

# --- Function to Format Final Output ---

def format_final_output(final_state: AppState) -> Dict[str, Any]:
    """
    Transforms the final AppState into the desired pipeline JSON structure.
    """
    segments_output = None
    if final_state.segments:
        # Use pydantic_encoder for robust serialization of list of Pydantic models
        segments_output = json.loads(json.dumps(final_state.segments, default=pydantic_encoder))
        # Alternative manual conversion:
        # segments_output = [segment.dict() for segment in final_state.segments]

    # Output structure of final object in Graph pipeline
    pipeline_output = {
        "status": "success" if not final_state.error else "error",
        "error_message": final_state.error,
        "results": {
            "proofread_text": final_state.proofread_text,
            "quality_score": final_state.metadata.get("quality_score"), # Safely get score
            "segments": segments_output,
            "original_char_count": final_state.metadata.get("initial_char_count"),
            "original_word_count": final_state.metadata.get("initial_word_count"),
            "proofread_char_count": final_state.metadata.get("proofread_length"),
            # Add any other specific metadata your pipeline needs
        },
        # Optionally include selected metadata if needed by the pipeline
        # "processing_details": {
        #     "segmentation_successful": final_state.metadata.get("segmentation_successful"),
        #     "proofreading_successful": final_state.metadata.get("proofreading_successful"),
        # }
    }
    return pipeline_output


if __name__ == "__main__":
    print("--- Running Text Processing Workflow ---")

    example_text = """
    The old house stood on a hill overlooking the town. Inside, shadows danced.
    "Is anyone there?" whispered Elias, his voice trembling slightly. Footsteps echoed from upstairs.
    Narrator: He clutched the rusty key tighter.
    A voice, cold and distant, replied, "We have been waiting." it said
    Elias: "Waiting for what?". He wishd he hadnt come.
    """

    app = create_workflow()
    initial_state = AppState(input_text=example_text)

    print("\nInvoking workflow...")
    config = {"recursion_limit": 50}
    final_state_dict = app.invoke(initial_state, config=config)

    # Parse the final state dictionary back into an AppState object for easy access
    final_state_obj = AppState.parse_obj(final_state_dict)

    print("\n--- Workflow Execution Complete ---")

    # --- Format the output for the pipeline ---
    print("\n--- Generating Pipeline Output JSON ---")
    pipeline_json_output = format_final_output(final_state_obj)

    # Pretty print the final JSON output
    print(json.dumps(pipeline_json_output, indent=2))

    # Save the JSON output to a file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"text_processing_output_{timestamp}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(pipeline_json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_filename}")
    print("\n--- End of Report ---")

