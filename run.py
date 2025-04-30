

from segmentation_graph import *
load_dotenv()


def main():
    """  
    Example of a run of the segmentation.py pipeline to:  
    1. Segment input text into speaker-text pairs.  
    2. Proofread the text and compute metadata (e.g., quality scores, counts).  

    Returns a JSON with:  
    - **Processed segments** (speaker-labeled text chunks).  
    - **Proofread results** (corrected text, quality metrics).  
    - **Metadata** (character/word counts, status flags).  

    Output Format:  
    ```json
    {
        "status": "success" | "error",  # Pipeline status  
        "error_message": str | null,    # Error details if failed  
        "results": {  
            "proofread_text": str,      # Final corrected text  
            "quality_score": float,     # 0-1 score (higher = better)  
            "segments": [               # Speaker-text pairs  
                {  
                    "speaker": str,     # "[Speaker_Name]"  
                    "text": str        # Segment content  
                },  
                ...  
            ],  
            "original_char_count": int, # Input text length (chars)  
            "original_word_count": int, # Input text length (words)  
            "proofread_char_count": int # Proofread text length (chars)  
        }  
    }
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

    print(json.dumps(pipeline_json_output, indent=2))
    # Save the JSON output to a file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"text_processing_output_{timestamp}.json"

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(pipeline_json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_filename}")
    print("\n--- End of Report ---")

if __name__ == "__main__":
    main()
