import json
from pydantic import BaseModel, Field, ValidationError
from typing import List
from vertexai.generative_models import GenerativeModel
from src.utils import retry_with_backoff, clean_json
import asyncio
import logging


class SearchQuery(BaseModel):
    section: str = Field(..., description="The section of the outline the query corresponds to.")
    query: str = Field(..., description="The search query for this section.")


async def generate_search_query_response(outline: str, model: GenerativeModel, logger: logging.Logger = None) -> str:
    """
    Generate a raw JSON response containing search queries based on the provided outline.

    Args:
        outline (str): The detailed outline of the article to generate queries for.
        model (GenerativeModel): The generative model instance used to produce the queries.

    Returns:
        str: The raw JSON response from the model (as a string).
    """
    prompt = f"""
    You are tasked with generating search queries to find corroborating evidence for key claims in a knowledgebase article.
    The goal is to identify relevant scientific papers to support and enhance the article, ensuring credibility and depth.

    TASK:
    - Review the provided outline of the knowledgebase article.
    - Identify areas or claims that would benefit from further evidence or scientific backing.
    - For each identified section, create a search query targeting relevant scientific papers or data.

    REQUIREMENTS:
    1. Return the search queries in strict JSON format.
    2. Each query must include:
        - 'section': The section of the outline the query corresponds to.
        - 'query': A specific search term designed to find relevant papers or abstracts.
    3. Use simple, standalone search terms or phrases. Avoid logical operators like `AND`, `OR`, or quotation marks.

    GUIDELINES:
    - Tailor queries to address gaps in evidence or provide additional insights for key claims in the article.
    - Ensure search terms are specific enough to yield meaningful results.
    - Avoid overly generic queries that may return irrelevant data.

    OUTPUT FORMAT:
    - Return a list of JSON objects, with each object containing the fields 'section' and 'query'.
    - Example:
        [
            {{"section": "Overview", "query": "Global impact of mosquito-borne diseases"}},
            {{"section": "Symptoms", "query": "Large local reactions to mosquito bites and immune response"}}
        ]

    ARTICLE OUTLINE:
    {outline}

    IMPORTANT:
    - Focus on generating precise and targeted queries to find corroborating evidence.
    - Do not include additional commentary or responses outside the JSON format.
    - Output the response as strict JSON without any additional commentary or formatting. Do not include code block markers like triple backticks (`).
    """

    async def call_generate_content():
        response = await model.generate_content_async(prompt)
        response_text = clean_json(response.text)  # Extract the response text
        return parse_search_queries(response_text)  # Parse after generation
    
    return await retry_with_backoff(call_generate_content, logger=logger)


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
        
        response_text = response_text.strip()
        
        # Strip potential code block markers (if the model includes triple backticks)
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text.strip("```json").strip("```").strip()

        # Parse the JSON response into a Python structure
        queries_json = json.loads(response_text)

        # Validate each query using Pydantic
        queries = [SearchQuery(**query) for query in queries_json]

        return queries
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ValueError(f"Error parsing or validating search queries: {e}\nResponse text: {response_text}")
