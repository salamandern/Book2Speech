

from segmentation_graph import *



load_dotenv()


def main():
    """
    Main function to create and run the langgraph graph
    """
    app = create_workflow()

    text_path = r"texts\john_snow_chapter.txt"

    input_text = open(text_path, "r").read()
    
    input_state = {
        "input_text": input_text
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