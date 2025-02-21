# Book2Speech

Transform any written story with different characters into an audiobook that you can listen to while doing other activities.

## Overview

Book2Speech is a Python-based application that uses Large Language Models (LLMs) and Langgraph to automatically process written stories into segmented, character-based audio content. It intelligently identifies different speakers and narrative sections, making it perfect for creating dynamic audiobook experiences.

## Features

- Automatic text segmentation by speaker/narrator
- Character identification and attribution
- Quality assurance through proofreading
- Automated workflow using Langgraph
- Quality scoring for output validation

## How it Works

The application uses a Langgraph-based LLM pipeline with several key stages:

1. **Supervision**: Initial text validation and metadata collection
2. **Segmentation**: Identifies and separates text by speaker/narrator
3. **Proofreading**: Ensures clarity and coherence of segmented text
4. **Quality Check**: Validates the final output quality

## Prerequisites

- Python 3.8+
- Google Gemini API access
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
2. Create a .env file and store the Gemini API key under the following variable name GEMINI_API_KEY= xxx
3. Change the text in the input state in the prototype code
4. Run the python file