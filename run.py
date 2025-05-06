from segmentation_graph import *



load_dotenv()


def main():
    """
    Main function to test the Book2speech code
    """
    
    app = create_workflow()
    initial_state = AppState(input_text=example_text)
    print("\nInvoking workflow...")
    config = {"recursion_limit": 50}
    final_state_dict = app.invoke(initial_state, config=config)
    final_state_obj = AppState.parse_obj(final_state_dict)
    print("\n--- Workflow Execution Complete ---")
    print("\n--- Generating Pipeline Output JSON ---")
    pipeline_json_output = format_final_output(final_state_obj)
    print(json.dumps(pipeline_json_output, indent=2))

    

if __name__ == "__main__":
    main()
