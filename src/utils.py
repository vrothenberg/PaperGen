import os
import random
import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, TypeVar, ParamSpec

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
    max_retries: int = 5,
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
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                if logger:
                    logger.info(f"[Retry {attempt + 1}/{max_retries}] Function {func.__name__} failed. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"[Error] Function {func.__name__} failed after {max_retries} retries.")
                raise e
            

def clean_json(text: str) -> str:
    """
    Cleans JSON data potentially wrapped with ```json ...` markers.
    """
    text = text.strip()  # Remove leading/trailing whitespace
    if text.startswith("```json") and text.endswith("```"):
        return text[7:-3].strip()  # Remove ```json and ```, then strip again for newlines
    elif text.startswith("`json") and text.endswith("`"): # handle single backticks as well
        return text[5:-1].strip()
    return text