import asyncio
import json
from dotenv import load_dotenv
load_dotenv()
from enrichment.state import InputState
from enrichment.configuration import Configuration
from enrichment import graph  # Assumes your simplified agent code is in my_agent.py

async def main():
    # In this example, we repurpose the 'topic' field to hold the input text.
    input_text = (
        "OpenAI was founded by Sam Altman and others. "
        "Their website is https://openai.com. "
        "They offer AI products such as ChatGPT, DALL-E, and Codex."
    )

    # Define the extraction schema for company information.
    extraction_schema = {
        "type": "object",
        "properties": {
            "founder": {
                "type": "string",
                "description": "The name of the company founder."
            },
            "websiteUrl": {
                "type": "string",
                "description": "Website URL of the company, e.g.: https://openai.com/ or https://microsoft.com"
            },
            "products_sold": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of products sold by the company."
            },
        },
        "required": ["founder", "websiteUrl", "products_sold"],
    }

    # Create the initial InputState.
    # Note: Here, 'topic' holds the text that the agent will process.
    initial_state = InputState(
        topic=input_text,
        extraction_schema=extraction_schema
    )

    # Set up the configuration.
    # The prompt now instructs the assistant to extract info directly from the given text.
    config_instance = Configuration(
        model="openai/gpt-4o",
        prompt=(
            "You are an assistant tasked with extracting specific information from the provided text using the extraction schema.\n\n"
            "Schema:\n{info}\n\n"
            "Text:\n{topic}\n\n"
            "Please provide your answer directly in clear text, filling in the schema."
        ),
        max_loops=3
    )

    # Convert the configuration to a dictionary.
    config = config_instance.__dict__

    # Run the workflow graph.
    final_state = await graph.ainvoke(initial_state, config)

    # Print the final extracted information (using dict indexing).
    print("Final Extracted Information:")
    print(json.dumps(final_state["info"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
