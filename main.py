import os
import json
import logging
from typing import List, Dict

import asyncio
from rich.logging import RichHandler
from google.generativeai import GenerativeModel

from src.config import SEMANTIC_SCHOLAR_API_KEY
from src.generators import (
    generate_outline,
    refine_outline_with_uptodate,
    integrate_papers,
    generate_search_query_response,
)
from src.pubmed import PubMedAPI
from src.semanticscholar import SemanticScholarAPI
from src.utils import save_results

# Constants
OUTPUT_DIR = "data/output"
SEARCH_RESULTS_FILE = "data/search_results.json"
LOG_FILE_NAME = "pipeline.log"
MAX_CONCURRENT_TASKS = 10
GENERATIVE_MODEL_NAME = "gemini-1.5-pro"
TOP_UNIQUE_FILES = 3


def setup_logger(output_dir: str) -> logging.Logger:
    """Sets up the logger to log to both the console and a file."""
    log_file = os.path.join(output_dir, LOG_FILE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("PaperGenerationPipeline")
    logger.setLevel(logging.INFO)

    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    # Console Handler
    console_handler = RichHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def extract_top_unique_files(results: List[Dict], top_n: int = TOP_UNIQUE_FILES) -> List[str]:
    """
    Extracts the top N unique markdown file paths from the results.

    Args:
        results (List[Dict]): List of result dictionaries with 'path' and 'score' keys.
        top_n (int): Number of top unique files to return.

    Returns:
        List[str]: List of top N unique file paths.
    """
    seen = set()
    unique_files = []

    for item in results:
        path = item.get("path")
        score = item.get("score")
        if path and score is not None:
            file_name = os.path.basename(path)
            key = (file_name, score)
            if key not in seen:
                seen.add(key)
                unique_files.append(path)
                if len(unique_files) == top_n:
                    break

    return unique_files


async def read_markdown_files(uptodate_paths: List[str], logger: logging.Logger) -> str:
    """Reads and concatenates multiple markdown files for integration."""
    combined_content = []
    for path in uptodate_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content.append(content)
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}", exc_info=True)
    spacer = "-"*80
    return f"\n{spacer}\n".join(combined_content)


async def process_topic(
    index: int,
    topic_query: Dict,
    model: GenerativeModel,
    pubmed_client: PubMedAPI,
    semantic_client: SemanticScholarAPI,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """
    Processes a single topic query asynchronously.

    Args:
        index (int): Index of the topic.
        topic_query (Dict): Dictionary containing the topic query.
        model (GenerativeModel): Generative AI model.
        pubmed_client (PubMedAPI): PubMed API client.
        semantic_client (SemanticScholarAPI): Semantic Scholar API client.
        output_dir (str): Directory to save outputs.
        logger (logging.Logger): Logger for logging information.

    Returns:
        str: Final integrated article.
    """
    try:
        topic = topic_query.get("query", "Untitled Topic")
        logger.info(f"[{index + 1}] Processing topic: {topic}")

        # Generate Outline
        outline = await generate_outline(topic, model)
        topic_dir = os.path.join(output_dir, topic.replace(' ', '_'))
        os.makedirs(topic_dir, exist_ok=True)
        save_results(outline, f"{topic}_outline", topic_dir, logger)

        # Integrate UpToDate Content if available
        results = topic_query.get("results", [])
        if results:
            unique_files = extract_top_unique_files(results)
            json_paths = [
                path.replace('.md', '.json') for path in unique_files
                if os.path.exists(path.replace('.md', '.json'))
            ]

            if json_paths:
                logger.info(f"[{index + 1}] Integrating UpToDate content for: {topic}")
                uptodate_content = await read_markdown_files(json_paths, logger)
                outline = await refine_outline_with_uptodate(
                    topic, outline, uptodate_content, model, logger
                )
                save_results(outline, f"{topic}_uptodate", topic_dir, logger)
            else:
                logger.warning(f"[{index + 1}] No valid UpToDate JSON files found for: {topic}")
        else:
            logger.warning(f"[{index + 1}] No results found in topic query.")

        # Generate Search Queries and Fetch Papers
        search_queries = await generate_search_query_response(outline, model, logger)
        parsed_queries = [q.model_dump() for q in search_queries]

        pubmed_task = pubmed_client.query(parsed_queries)
        semantic_task = semantic_client.query(parsed_queries)

        pubmed_results, semantic_results = await asyncio.gather(pubmed_task, semantic_task)

        formatted_pubmed = pubmed_client.format_results(pubmed_results)
        formatted_semantic = semantic_client.format_results(semantic_results)

        papers = formatted_pubmed + formatted_semantic
        papers_json = json.dumps(papers, separators=(',', ':'), ensure_ascii=False)
        save_results(papers_json, f"{topic}_papers", topic_dir, logger)
        logger.info(f"[{index + 1}] Integrated {len(papers)} papers for: {topic}")

        # Integrate Papers into Final Article
        final_article = await integrate_papers(topic, outline, papers, model, logger)
        save_results(final_article, f"{topic}_final", topic_dir, logger)
        logger.info(f"[{index + 1}] Finished processing topic: {topic}")

        return final_article

    except Exception as e:
        logger.error(f"[{index + 1}] Error processing topic '{topic}': {e}", exc_info=True)
        return ""


async def main():
    """Main function to execute the paper generation pipeline."""
    try:
        # Setup
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger = setup_logger(OUTPUT_DIR)
        logger.info("Starting Paper Generation Pipeline...")

        # Load Search Results
        logger.info(f"Loading search results from '{SEARCH_RESULTS_FILE}'...")
        try:
            with open(SEARCH_RESULTS_FILE, 'r', encoding='utf-8') as file:
                topics_data = json.load(file)
            logger.info(f"Loaded {len(topics_data)} topics.")
        except FileNotFoundError:
            logger.critical(f"Search results file '{SEARCH_RESULTS_FILE}' not found.")
            return
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON format in '{SEARCH_RESULTS_FILE}': {e}")
            return

        # Initialize API Clients and Model
        pubmed_client = PubMedAPI(api_key=None)  # Replace with actual API key if needed
        semantic_client = SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY)
        model = GenerativeModel(GENERATIVE_MODEL_NAME)

        # Concurrency Control
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        tasks = []

        for idx, topic in enumerate(topics_data):
            async def bound_process(idx=idx, topic=topic):
                async with semaphore:
                    return await process_topic(
                        idx,
                        topic,
                        model,
                        pubmed_client,
                        semantic_client,
                        OUTPUT_DIR,
                        logger
                    )
            tasks.append(bound_process())

        # Execute Tasks
        await asyncio.gather(*tasks)
        logger.info("Paper Generation Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Critical error in the pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
