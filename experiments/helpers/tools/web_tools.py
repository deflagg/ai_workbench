import json
import os
from langchain_core.tools import tool
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

@tool
def google_search(search_term: str, num_results: int = 10) -> list:
    """
    Performs a Google Custom Search and returns the results.

    Args:
        search_term: The query to search for.
        num_results: The maximum number of search results to return (default is 10).

    Returns:
        A list of dictionaries, each containing information about a search result 
        (title, link, snippet).

    Raises:
        ValueError: If the GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables are not set.
        HttpError: If there's an error communicating with the Google API.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set as environment variables.")

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(
            q=search_term, cx=cse_id, num=num_results
        ).execute()

        search_results = res.get("items", [])
        return search_results
    except HttpError as error:
        raise HttpError(f"An error occurred: {error}")

if __name__ == "__main__":
    search_term = "LangChain"
    os.environ["GOOGLE_CSE_ID"] = "b6aafeb3d35b1441a" # search engine id
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA6x1nMWJFgHlgiwp2Ci1ylELL9y4Kw0HM"
    results = google_search(search_term)

    # Pretty-print the results
    print(json.dumps(results, indent=2))
