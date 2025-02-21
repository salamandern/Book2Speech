from typing import Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
from pydantic import BaseModel
from typing import Dict, List, Optional


load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class Segment(BaseModel):
    speaker: str
    text: str

class AppState(BaseModel):
    input_text: str
    segments: Optional[List[Segment]] = None
    proofread_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    


def create_gemini_model():
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    return model

def supervisor_node(state: AppState) -> Dict:
    """
    Supervises the workflow and performs initial validation.
    
    Args:
        state: Current application state
        
    Returns:
        Dictionary with any updates for downstream nodes
    """
    print(f"Supervisor processing text of length: {len(state.input_text)}")
    
    if not state.input_text.strip():
        raise ValueError("Input text cannot be empty")
    
    # Update metadata in state
    state.metadata.update({
        "char_count": len(state.input_text),
        "word_count": len(state.input_text.split())
    })
    
    return {
        "status": "validated", 
        "metadata": state.metadata
    }

def segment_agent(state: AppState) -> Dict:
    """
    Segments the input text into meaningful chunks with speaker attribution
    
    Args:
        state: Current application state containing input text
        
    Returns:
        Dictionary containing the segmented text with speaker information
    """
    model = create_gemini_model()

    
    segment_prompt = f"""
    Please segment the following text by identifying who is speaking in each part.
    Identify if it's a character speaking or a narrator describing the scene.
    
    Text to segment:
    {state.input_text}
    
    Return the output in this exact JSON format:
    {{
        "segments": [
            {{
                "speaker": "character_name or narrator",
                "text": "the spoken text or narration"
            }}
        ]
    }}

    Return ONLY the JSON with no markdown formatting or additional text.
    """
    
    response = model.generate_content(segment_prompt)

    response_text = response.text

    if response_text.startswith("```"):
        print("Still doing the same shit huh")
        match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        response_text = match.group(1)

    print("!!!! REPONSE ")
    print(response_text)
    
    try:
        # Parse the response as JSON
        parsed_response = json.loads(response_text)
        
        segments = [
            Segment(speaker=seg["speaker"], text=seg["text"])
            for seg in parsed_response["segments"]
        ]
                
        state.segments = segments
        print(f"Text segmented into {len(segments)} chunks")
        
        # Update metadata in state
        state.metadata.update({
            "num_segments": len(segments),
            "segmentation_successful": True
        })
        
        # Return in the expected format
        return {
            "segments": [
                {
                    "speaker": segment.speaker,
                    "text": segment.text
                }
                for segment in segments
            ],
            "metadata": state.metadata
        }
        
    except (json.JSONDecodeError, KeyError) as e:
        # Update metadata for error case
        state.metadata.update({
            "segmentation_error": str(e),
            "segmentation_successful": False
        })
        return {
            "error": "Failed to process segments",
            "metadata": state.metadata
        }

def proofread_agent(state: AppState) -> Dict:
    """
    Proofreads and corrects the segmented text using an llm 
    
    Args:
        state: Current application state
        
    Returns:
        Dictionary containing the proofread text
    """
    model = create_gemini_model()


    
    # Create a comprehensive proofreading prompt
    proofread_prompt = """
    Please proofread and improve the following text. Focus on:
    1. Clarity and coherence
    2. Correct character segmentation
    3. Correct narrator segmentation
    
    Text to proofread:
    {text}
    
    Return only the corrected text in the input format without explanations.
    """.format(text=" ".join([i.text for i in state.segments]))
    
    response = model.generate_content(proofread_prompt)
    
    state.proofread_text = response.text.strip()

    print("Current state: ", state)

    # Update metadata in state
    state.metadata.update({
        "corrections_made": True,
        "proofread_length": len(state.proofread_text)
    })

    return {
        "proofread_text": state.proofread_text,
        "metadata": state.metadata
    }

def quality_check_agent(state: AppState) -> Dict:
    """
    Performs a final quality check on the proofread text.
    
    Args:
        state: Current application state
        
    Returns:
        Dictionary containing quality metrics
    """
    model = create_gemini_model()
    print("Current state: ", state)

    
    quality_prompt = f"""
    Analyze the following text for quality and return a score from 1-10:
    
    {state.proofread_text}
    
    Return only the numerical score.
    """
    
    response = model.generate_content(quality_prompt)
    quality_score = float(response.text.strip())
    
    # Update metadata in state
    state.metadata.update({
        "quality_score": quality_score,
        "workflow_completed": True
    })
    
    return {
        "quality_score": quality_score,
        "metadata": state.metadata
    }

def create_workflow() -> StateGraph:
    """
    Creates and configures the workflow graph with all nodes and edges.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    graph = StateGraph(AppState)
    
    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("segmenter", segment_agent)
    graph.add_node("proofreader", proofread_agent)
    graph.add_node("quality_check", quality_check_agent)
    
    # Define the workflow
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "segmenter")
    graph.add_edge("segmenter", "proofreader")
    graph.add_edge("proofreader", "quality_check")
    graph.add_edge("quality_check", END)
    
    return graph.compile()

def main():
    """
    Main function to run the workflow with example input.
    """
    app = create_workflow()
    
    input_state = {
        "input_text": """
        The Picnic Plan

        Emma and Jake were best friends who loved spending time together. One sunny Saturday morning, Emma called Jake with an idea.

        "Let's go on a picnic!" she said excitedly.
        "That sounds perfect!" Jake replied. "I'll bring the sandwiches, and you bring the snacks?"
        "Deal!" Emma agreed.

        They met at the park an hour later. Emma carried a basket full of chips, fruit, and cookies, while Jake brought a cooler with sandwiches and lemonade. They found a cozy spot under a big oak tree and spread out a checkered blanket.

        As they ate, they laughed about old memories and made plans for the summer. Suddenly, a gust of wind blew Jake's hat off his head and sent it tumbling toward the pond.

        "Oh no!" Jake cried, jumping up.
        Emma giggled. "I'll get it!" She ran after the hat, but just as she reached for it, she slipped on the wet grass and landed with a splash in the shallow water.

        Jake burst out laughing. "Well, I guess the hat's not the only thing that needed a swim!"
        Emma stood up, dripping wet but smiling. "At least I saved your hat!" she said, holding it up triumphantly.

        They both laughed until their sides hurt. Even though the picnic didn't go exactly as planned, it turned into a day they'd never forget.
        """
    }
    
    try:
        final_state = app.invoke(input_state)
        print("\nWorkflow Results:")
        print("----------------")

        #print(f"Original Text: {final_state.input_text}")
        print(f"Segments: {final_state["segments"]}")
        print(f"Final Proofread Text: {final_state["proofread_text"]}")
        # print(f"Quality Score: {final_state["metadata"].get('quality_score')}")
        print(f"Metadata: {final_state["metadata"]}")
    except Exception as e:
        print(f"Error during workflow execution: {str(e)}")
        raise e

if __name__ == "__main__":
    main()