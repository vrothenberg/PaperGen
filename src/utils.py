# src/utils.py
import os
import re
import json
import random
import hashlib
import asyncio
import logging
from datetime import datetime
from typing import Any, List, Dict, Callable, Coroutine, TypeVar, ParamSpec, Union, Set
from src.models import SearchQuery, Article, Paper
from pydantic import ValidationError

T = TypeVar('T')
P = ParamSpec('P')


def save_results(
    index:int, data, topic: str, output_dir: str, logger: logging.Logger = None
) -> None:
    """
    Saves the given article content to a markdown file in the specified output directory.

    Args:
        article (str): The article content to save.
        topic (str): The name of the topic associated with the article.
        output_dir (str): The directory where the file will be saved.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitize_filename(topic)}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(data)
        logger.info(f"[{index+1}] Results saved to: {filepath}")
    except Exception as e:
        logger.error(f"[{index+1}] Error writing to: {filepath}")
        logger.error(e)


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


def sanitize_filename(dir):
    # Replace spaces with underscores and remove all non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^\w\s]', '', dir).replace(' ', '_')
    return sanitized


def validate_article_json(article_json: str, logger: logging.Logger) -> bool:
    """
    Validates the article JSON against the Pydantic Article model.

    Args:
        article_json (str): The article JSON as a string.
        logger (logging.Logger): Logger for logging information.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        article_dict = json.loads(article_json)
        article = Article(**article_dict)
        logger.debug("Article JSON is valid according to the Article model.")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return False
    except ValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return False


def expand_citation_ranges(citation: str) -> List[str]:
    """
    Expand a citation range like [2,23-25] into a list of individual numbers: ['2', '23', '24', '25'].
    """
    expanded = []
    for part in citation.split(","):
        if "-" in part:  # Handle ranges like 23-25
            range_parts = part.split("-")
            if len(range_parts) == 2:  # Ensure it's a valid range
                try:
                    start, end = map(int, range_parts)
                    expanded.extend(map(str, range(start, end + 1)))
                except ValueError:
                    print(f"Invalid range format: {part}. Skipping.")
                    continue
            else:
                print(f"Unexpected range format: {part}. Skipping.")
        else:  # Single reference like 2
            try:
                expanded.append(part.strip())
            except ValueError:
                print(f"Invalid reference: {part}. Skipping.")
    return expanded


def extract_citations(line: str) -> List[str]:
    """
    Extract and expand citations from a line, ensuring they are valid numeric references.
    """
    citations = []
    for match in re.findall(r"\[(.*?)\]", line):  # Find content inside square brackets
        # Ensure the match contains only numbers, commas, or hyphens
        if re.match(r"^\d+([,-]\d+)*$", match):
            citations.extend(expand_citation_ranges(match))
    return citations


def remove_duplicates(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate results based on the 'content' field.
    """
    seen = set()
    unique_results = []
    for result in results:
        content_hash = hashlib.md5(result["content"].encode('utf-8')).hexdigest()
        if content_hash not in seen:
            unique_results.append(result)
            seen.add(content_hash)
    return unique_results


def parse_reference(ref_str: str, chunk_title: str, logger: logging.Logger):
    """
    Parses a reference string to extract the reference number and citation.

    Args:
        ref_str (str): The reference string (e.g., "[24] De Rycke L, ...")
        chunk_title (str): Title of the chunk for logging purposes.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: A dictionary with 'reference_number' and 'citation' keys, or None if parsing fails.
    """
    match = re.match(r'\[(\d+)\]\s*(.*)', ref_str)
    if not match:
        logger.warning(f"Invalid reference format in chunk '{chunk_title}': {ref_str}")
        return None
    ref_number = int(match.group(1))
    citation = match.group(2).strip()
    return {
        'reference_number': ref_number,
        'citation': citation
    }







def get_bad_references(data: Dict) -> Set[int]:
    """Identifies references that are missing required fields."""
    bad_references = set()
    if "references" in data and "content" in data["references"]:
        for ref in data["references"]["content"]:
            if (
                not ref.get("journal_source")
                or not ref.get("authors")
                or not ref.get("year")
            ):
                bad_references.add(ref["reference_number"])
    return bad_references



def identify_all_inline_citations(data: Dict) -> Set[int]:
    """
    Identifies all unique inline citations (e.g., [1], [2, 3]) in the document.

    Recursively processes nested dictionaries and lists within all sections,
    except for the "references" section.
    """

    def _extract_citations_from_text(text: str) -> Set[int]:
        """Helper function to extract citations from a text string."""
        citations = set()
        matches = re.findall(r"\[([^\]]+)\]", text)
        for match in matches:
            for ref in match.split(","):
                ref = ref.strip()
                if ref.isdigit():
                    citations.add(int(ref))
        return citations

    def _recursive_process(item: Any, in_references_section: bool = False) -> Set[int]:
        """Recursively processes dictionaries, lists, and strings."""
        citations = set()
        if isinstance(item, str):
            if not in_references_section:
                citations.update(_extract_citations_from_text(item))
        elif isinstance(item, dict):
            # Correctly set in_references_section for child elements
            is_current_references = "references" in item
            for value in item.values():
                citations.update(
                    _recursive_process(value, is_current_references)
                )  # Pass the flag based on the current item

        elif isinstance(item, list):
            for list_item in item:
                citations.update(_recursive_process(list_item, in_references_section))
        return citations

    return _recursive_process(data)


def get_all_reference_numbers(references_section: Dict) -> Set[int]:
    """
    Extracts all reference numbers from the 'references' section,
    even if they are not integers (for finding unused references).
    """
    reference_numbers = set()
    if isinstance(references_section, dict):
        for item in references_section.get("content", []):
            ref_num = item.get("reference_number")
            if ref_num is not None:
                reference_numbers.add(ref_num)
    return reference_numbers




def remove_references(data: Dict, references_to_remove: set) -> Dict:
    """Removes specified references from the 'references' section."""
    if "references" in data and "content" in data["references"]:
        data["references"]["content"] = [
            ref
            for ref in data["references"]["content"]
            if ref["reference_number"] not in references_to_remove
        ]
    return data


def remove_inline_citations(data: Dict, references_to_remove: set) -> Dict:
    """Removes inline citations corresponding to the specified reference numbers.

    Recursively processes nested dictionaries and lists within all sections,
    except for the "references" section.
    """

    def _process_text(text: str) -> str:
        """Helper function to process individual text strings."""
        return re.sub(
            r"\[([^\]]+)\]",
            lambda match: "["
            + ",".join(
                ref.strip()
                for ref in match.group(1).split(",")
                if not (ref.strip().isdigit() and int(ref.strip()) in references_to_remove)
            )
            + "]"
            if any(
                ref.strip().isdigit() and int(ref.strip()) not in references_to_remove
                for ref in match.group(1).split(",")
            )
            else "",
            text,
        )

    def _recursive_process(item: Any, in_references_section: bool = False) -> Any:
        """Recursively processes dictionaries, lists, and strings.
        Skips processing if inside the 'references' section.
        """
        if isinstance(item, str):
            return item if in_references_section else _process_text(item)
        elif isinstance(item, dict):
            if "references" in item:
                return {
                    key: _recursive_process(value, key == "references")
                    for key, value in item.items()
                }
            else:
                return {
                    key: _recursive_process(value, in_references_section)
                    for key, value in item.items()
                }

        elif isinstance(item, list):
            return [_recursive_process(list_item, in_references_section) for list_item in item]
        else:
            return item

    return _recursive_process(data)


def create_remap_dictionary(data: Dict) -> Dict:
    """Creates a dictionary to map old reference numbers to new sequential ones."""
    reference_numbers = []
    if "references" in data and "content" in data["references"]:
        for ref in data["references"]["content"]:
            if "reference_number" in ref:
                reference_numbers.append(ref["reference_number"])

    reference_numbers.sort()
    return {
        old_num: new_num
        for new_num, old_num in enumerate(reference_numbers, start=1)
    }


def update_reference_numbers(data: Dict, remap: Dict) -> Dict:
    """Updates reference numbers in the 'references' section based on the remap dictionary."""
    if "references" in data and "content" in data["references"]:
        for ref in data["references"]["content"]:
            if "reference_number" in ref:
                ref["reference_number"] = remap[ref["reference_number"]]
    return data


def update_inline_citations(data: Dict, remap: Dict) -> Dict:
    """Updates inline citations to reflect new reference numbers based on the remap dictionary.

    Recursively processes nested dictionaries and lists within all sections,
    except for the "references" section.
    """

    def _process_text(text: str, remap: Dict) -> str:
        """Helper function to process individual text strings."""

        def _replace_citation(match):
            updated_refs = []
            for ref in match.group(1).split(","):
                ref_str = ref.strip()
                if ref_str.isdigit():
                    ref_int = int(ref_str)
                    updated_ref = remap.get(ref_int, ref_str)  # Use int key, fallback to original if not found
                else:
                    updated_ref = ref_str
                updated_refs.append(str(updated_ref))
            return "[" + ",".join(updated_refs) + "]"

        return re.sub(r"\[([^\]]+)\]", _replace_citation, text)

    def _recursive_process(item: Any, remap:Dict, in_references_section: bool = False) -> Any:
        """Recursively processes dictionaries, lists, and strings.
        Skips processing if inside the 'references' section.
        """
        if isinstance(item, str):
            return item if in_references_section else _process_text(item, remap)
        elif isinstance(item, dict):
            if "references" in item:
                return {
                    key: _recursive_process(value, remap, key == "references")
                    for key, value in item.items()
                }
            else:
                return {
                    key: _recursive_process(value, remap, in_references_section)
                    for key, value in item.items()
                }
        elif isinstance(item, list):
            return [
                _recursive_process(list_item, remap, in_references_section)
                for list_item in item
            ]
        else:
            return item

    return _recursive_process(data, remap)


def check_for_duplicate_references(data: Dict) -> bool:
    """Checks if the references section contains duplicate reference numbers"""
    reference_numbers = []
    if "references" in data and "content" in data["references"]:
        for ref in data["references"]["content"]:
            if "reference_number" in ref:
                reference_numbers.append(ref["reference_number"])

    # Check for uniqueness
    return len(reference_numbers) != len(set(reference_numbers))


def sort_references_ascending(data: Dict) -> Dict:
    """Sorts the references in the 'references' section by 'reference_number' in ascending order."""
    if "references" in data and "content" in data["references"]:
        data["references"]["content"].sort(key=lambda x: x.get("reference_number", float('inf')))
    return data

def clean_references(data: Dict) -> Dict:
    """
    Cleans up references in the document:

    1. Identifies and removes orphaned inline citations.
    2. Identifies and removes unused references.
    3. Renumbers references and updates inline citations.
    """

    # 1. Identify Inline Citations and Reference Numbers
    inline_citations = identify_all_inline_citations(data)
    all_reference_numbers = get_all_reference_numbers(data.get("references", {}))

    # 2. Find Orphaned and Unused
    orphaned_citations = inline_citations - all_reference_numbers
    unused_references = all_reference_numbers - inline_citations

    # 3. Identify bad references
    bad_references = get_bad_references(data)

    # 4. Remove Orphaned Citations and Unused References
    data = remove_inline_citations(data, orphaned_citations)
    data = remove_references(data, unused_references)
    data = remove_references(data, bad_references)
    data = remove_inline_citations(data, bad_references)

    # 5. Renumber and Update
    remap_dict = create_remap_dictionary(data)
    data = update_reference_numbers(data, remap_dict)
    data = update_inline_citations(data, remap_dict)

    # 6. Check for Duplicates
    if check_for_duplicate_references(data):
        print("Warning: Duplicate reference numbers found after cleaning.")

    return data