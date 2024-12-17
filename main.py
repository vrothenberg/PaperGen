import os
import json
import logging
from rich.logging import RichHandler
import asyncio
from vertexai.generative_models import GenerativeModel
from typing import List, Dict

from src.config import SEMANTIC_SCHOLAR_API_KEY
from src.generators import generate_outline, refine_outline_with_uptodate, integrate_papers
from src.queries import generate_search_query_response
from src.pubmed import PubMedAPI
from src.semanticscholar import SemanticScholarAPI
from src.utils import save_results


def setup_logger(output_dir: str) -> logging.Logger:
    """Sets up the logger to log to both the console and a file."""
    log_file = os.path.join(output_dir, "pipeline.log")
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("PaperGenerationPipeline")
    logger.setLevel(logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    console_handler = RichHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


async def process_topic_query(
    i: int,
    topic_query: dict,
    model: GenerativeModel,
    pubmed_client: PubMedAPI,
    semantic_scholar_client: SemanticScholarAPI,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """Asynchronously processes a single topic query."""
    try:
        topic = topic_query['query']
        logger.info(f"Processing: {topic}")
        outline = await generate_outline(topic, model)
        topic_dir = os.path.join(output_dir, topic.replace(' ', '_'))
        os.makedirs(topic_dir, exist_ok=True)
        save_results(outline, topic + "_outline", topic_dir, logger)

        results = topic_query.get('results', [])
        if results:
            unique_files = extract_top_unique_files_from_json(results, top_n=5)
            uptodate_paths = [path.replace('.md', '.json') for path in unique_files if path.replace('.md','.json') and os.path.exists(path.replace('.md','.json'))]

            if uptodate_paths:
                logger.info(f"Integrating UpToDate for {topic}")
                uptodate_markdown = await read_multiple_uptodate_markdown(uptodate_paths, logger)
                outline = await refine_outline_with_uptodate(topic, outline, uptodate_markdown, model, logger)
                save_results(outline, topic + "_uptodate", topic_dir, logger)
            else:
                logger.warning(f"No valid UpToDate JSON files found for {topic}.")
        else:
            logger.warning("No results found in topic query.")


        queries = await generate_search_query_response(outline, model, logger)
        parsed_queries = [q.model_dump() for q in queries]
        pubmed_task = pubmed_client.query(parsed_queries)
        semantic_task = semantic_scholar_client.query(parsed_queries)
        pubmed_results, semantic_results = await asyncio.gather(pubmed_task, semantic_task)

        pubmed_results = pubmed_client.format_results(pubmed_results)
        semantic_results = semantic_scholar_client.format_results(semantic_results)
        
        papers = pubmed_results + semantic_results
        papers_string = json.dumps(papers, separators=(',', ':'), ensure_ascii=False)
        save_results(papers_string, topic + "_papers", topic_dir, logger)
        logger.info(f"Integrating {len(papers)} papers for {topic}")
        final_article = await integrate_papers(topic, outline, papers, model, logger)
        save_results(final_article, topic + "_final", topic_dir, logger)
        logger.info(f"Finished processing: {topic}")
        return final_article
    except Exception as e:
        logger.error(f"Error processing topic {i+1}: {e}", exc_info=True)
        return None

def extract_top_unique_files_from_json(results: List[Dict], top_n: int = 5) -> List[str]:
    """
    Extracts the top N unique markdown files from a list of results dictionaries.
    Uniqueness is determined by the markdown file name and score.
    It preserves the full path to the file.

    Args:
        results (List[Dict]): List of result dictionaries with 'path' and 'score' keys.
        top_n (int): Number of top unique files to return.

    Returns:
        List[str]: List of top N unique file paths.
    """
    seen_files = {}
    unique_files = []
    for item in results:
        path = item.get('path')
        score = item.get('score')
        if path and score is not None:
            file_name = os.path.basename(path)
            key = (file_name, score)
            if key not in seen_files:
              seen_files[key] = True
              unique_files.append(path)
    return unique_files[:top_n]


async def read_multiple_uptodate_markdown(uptodate_paths: List[str], logger: logging.Logger) -> str:
    """Reads and concatenates multiple markdown files for integration."""
    combined_content = []
    for path in uptodate_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content.append(content)
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}", exc_info=True)
    return "\n\n".join(combined_content)


async def main():
    """Main function to run the pipeline."""
    try:
        output_dir = "data/output"  # Adjust as needed
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger(output_dir)
        logger.info("Loading data from 'search_results.json'...")
        with open("data/search_results.json", 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} topics.")
        pubmed_client = PubMedAPI(api_key=None)
        semantic_scholar_client = SemanticScholarAPI(api_key=SEMANTIC_SCHOLAR_API_KEY)
        model = GenerativeModel("gemini-1.5-pro-002")
        max_concurrent_tasks = 3
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def process_with_semaphore(i, topic_query):
            async with semaphore:
                logger.info(f"Starting processing for topic {i+1}: {topic_query['query']}")
                result = await process_topic_query(
                    i, topic_query, model, pubmed_client, semantic_scholar_client, output_dir, logger
                )
                logger.info(f"Finished processing for topic {i+1}: {topic_query['query']}")
                return result
        tasks = [process_with_semaphore(i, topic_query) for i, topic_query in enumerate(data)]
        await asyncio.gather(*tasks)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Critical error in main pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())