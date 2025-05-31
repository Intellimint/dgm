import logging
import os
from llm import get_response_from_llm, create_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sysprompt():
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Create a client (using Claude as it's most reliable)
    client, model = create_client("claude-3-5-sonnet-20240620")
    
    # Test message
    test_msg = "Please confirm you received the Atlas system prompt by responding with 'Atlas system prompt received'."
    
    # Get response without explicitly providing system_message
    response, _ = get_response_from_llm(
        msg=test_msg,
        client=client,
        model=model,
        print_debug=True
    )
    
    print("\nResponse:", response)

if __name__ == "__main__":
    test_sysprompt() 