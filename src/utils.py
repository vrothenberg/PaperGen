import os
import re
import json
import random
import asyncio
import logging
from datetime import datetime
from typing import Any, List, Callable, Coroutine, TypeVar, ParamSpec
from src.models import SearchQuery

T = TypeVar('T')
P = ParamSpec('P')


def save_results(article: str, topic: str, output_dir: str, logger: logging.Logger = None) -> None:
    """
    Saves the given article content to a markdown file in the specified output directory.

    Args:
        article (str): The article content to save.
        topic (str): The name of the topic associated with the article.
        output_dir (str): The directory where the file will be saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{topic.replace(' ', '_')}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(article)
    logger.info(f"Results saved to: {filepath}")


async def retry_with_backoff(
    func: Callable[P, Coroutine[Any, Any, T]],
    max_retries: int = 10,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    logger: logging.Logger = None,
    *args: P.args,
    **kwargs: P.kwargs
) -> T:
    """
    Retry an asynchronous function with exponential backoff.

    Args:
        func (Callable): The async function to retry.
        max_retries (int): Maximum number of retries before failing.
        base_delay (float): Initial delay between retries in seconds.
        max_delay (float): Maximum delay between retries in seconds.
        logger (logging.Logger, optional): Logger object to log retries. Defaults to None.
        *args (P.args): Positional arguments to pass to the function.
        **kwargs (P.kwargs): Keyword arguments to pass to the function.

    Returns:
        T: The result of the function call if successful.

    Raises:
        Exception: The last exception encountered after exhausting retries.
    """
    for attempt in range(max_retries):
        try:
            response = await func(*args, **kwargs)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                if logger:
                    try:
                        # Attempt to extract usage_metadata and finish_reason
                        usage_metadata = response.to_dict()['usage_metadata'] if response else None
                        finish_reason = response.to_dict()['candidates'][0]['finish_reason'] if response else None
                        logger.info(f"[Retry {attempt + 1}/{max_retries}] Function {func.__name__} failed (reason: {finish_reason if finish_reason else 'Unknown'}). Retrying in {delay:.2f} seconds. Usage Metadata: {usage_metadata}")
                    except (KeyError, AttributeError):
                        logger.info(f"[Retry {attempt + 1}/{max_retries}] Function {func.__name__} failed. Retrying in {delay:.2f} seconds. Error: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"[Error] Function {func.__name__} failed after {max_retries} retries.")
                raise e


def clean_json(text: str) -> str:
    """
    Cleans JSON data by removing markdown code block delimiters and attempting to fix common JSON formatting issues.

    This function removes surrounding markdown code fences (```json```) or single backticks (`json`)
    from a JSON string if present. It also attempts to fix common issues like missing commas or brackets.

    Args:
        text (str): The input string potentially wrapped with markdown code blocks.

    Returns:
        str: The cleaned and potentially corrected JSON string without markdown delimiters.
    """
    # Remove leading and trailing whitespace
    cleaned_text = text.strip()

    # Remove code fences if present
    patterns = [
        r'^```json\s*\n?(.*?)\n?```$',  # Triple backticks with optional newlines
        r'^`json\s*(.*?)`$',             # Single backticks
    ]

    for pattern in patterns:
        match = re.match(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()
            break  # Exit after the first successful match

    # Attempt to fix common JSON issues
    # Example: Replace single quotes with double quotes
    cleaned_text = cleaned_text.replace("'", '"')

    # Ensure that all opening brackets have corresponding closing brackets
    open_brackets = cleaned_text.count('[') - cleaned_text.count(']')
    open_braces = cleaned_text.count('{') - cleaned_text.count('}')
    
    if open_brackets > 0:
        cleaned_text += ']' * open_brackets
    if open_braces > 0:
        cleaned_text += '}' * open_braces

    # Add missing commas between JSON objects in a list
    # This is a naive approach and may not handle all cases
    cleaned_text = re.sub(r'\}\s*\{', '},{', cleaned_text)

    return cleaned_text


def parse_search_queries(response_text: str) -> List[SearchQuery]:
    """
    Parse the raw JSON response from the model into a list of SearchQuery objects.

    Args:
        response_text (str): The raw JSON response from the model.

    Returns:
        List[SearchQuery]: A list of validated SearchQuery objects.

    Raises:
        ValueError: If the response cannot be parsed or validated.
    """
    try:
        if not response_text.strip():
            raise ValueError("Received empty response text.")

        # Clean the JSON string
        cleaned_text = clean_json(response_text)

        # Attempt to parse the JSON
        queries_json = json.loads(cleaned_text)

        # Validate each query using Pydantic
        queries = [SearchQuery(**query) for query in queries_json]

        return queries
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error: {e}\nResponse text: {cleaned_text}")
    except Exception as e:
        raise ValueError(f"Error parsing or validating search queries: {e}\nResponse text: {cleaned_text}")