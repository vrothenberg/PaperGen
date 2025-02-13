import os
import re
import json
import pickle
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional

import asyncio
from rich.logging import RichHandler
from google.generativeai import GenerativeModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import SEMANTIC_SCHOLAR_API_KEY
from src.generators import (
    generate_outline,
    refine_outline_with_uptodate,
    generate_search_query_response,
    filter_papers_by_relevancy,
    integrate_papers,
    comprehensive_edit
)
from src.semanticscholar import SemanticScholarAPI
from src.utils import (
    save_results,
    sanitize_filename,
    remove_duplicates,
    extract_citations,
    parse_reference,
    clean_references
)
from src.models import Article

# Constants
# GENERATIVE_MODEL_NAME = "gemini-1.5-flash"
# GENERATIVE_MODEL_NAME = "gemini-1.5-pro"
GENERATIVE_MODEL_NAME = "gemini-2.0-flash-exp"
OUTPUT_DIR = f"data/output/{GENERATIVE_MODEL_NAME}"
CSV_FILE = "data/condition_revised.csv"
# CSV_FILE = "data/missing_conditions.csv"
JOURNALS_FILE = "data/journals_df.csv"
SEARCH_RESULTS_FILE = "data/search_results_revised.json"
LOG_FILE_NAME = "pipeline.log"
MAX_CONCURRENT_TASKS = 5
TEMPERATURE = 1.0
TOP_UNIQUE_FILES = 20
MIN_SEARCH_SCORE = 10.0
MIN_CITATION_COUNT = 50
MIN_SJR_SCORE = 1.0


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


async def integrate_uptodate_content(
    index: int,
    condition_name: str,
    alternative_name: str,
    category: str,
    search_results: List[Dict],
    outline: Article,
    model: GenerativeModel,
    topic_dir: str,
    logger: logging.Logger,
) -> Article:
    """
    Integrate UpToDate content into the outline if available.

    Args:
        index (int): The index of the topic in the list.
        condition_name (str): Primary condition name.
        alternative_name (str): Alternative condition name.
        category (str): Category of the condition.
        search_results (List[Dict]): Search results for the condition.
        outline (Article): Current article outline as an Article object.
        model (GenerativeModel): LLM model.
        logger (logging.Logger): Logger instance.
        topic_dir (str): Directory for saving outputs.

    Returns:
        Article: Updated outline with integrated content.
    """

    try:
        # Step 1: Find relevant search results for condition and alternative name
        condition_result = next(
            (d for d in search_results if d.get('query') == condition_name), None
        )
        alternative_result = next(
            (d for d in search_results if d.get('query') == alternative_name), None
        )

        relevant_results = []
        if condition_result:
            unique_condition = remove_duplicates(condition_result['results'])
            filtered_condition = [
                r for r in unique_condition if r['score'] > MIN_SEARCH_SCORE
            ]
            relevant_results.extend(filtered_condition[:TOP_UNIQUE_FILES])

        if alternative_result:
            unique_alternative = remove_duplicates(alternative_result['results'])
            filtered_alternative = [
                r for r in unique_alternative if r['score'] > MIN_SEARCH_SCORE
            ]
            relevant_results.extend(filtered_alternative[:TOP_UNIQUE_FILES])

        if not relevant_results:
            logger.warning(
                f"[{index + 1}] No results found for queries: {condition_name}, {alternative_name}"
            )
            return outline 

        # Step 2: Extract existing reference numbers from the Outline's ReferencesSection
        existing_reference_numbers = {
            ref.reference_number for ref in outline.references.content
        }
        next_ref_number = (
            max(existing_reference_numbers) + 1 if existing_reference_numbers else 1
        )

        # Prepare to collect all new references
        all_new_references = []

        # List to hold updated content chunks
        uptodate_content_chunks = []

        # Step 3: Process each unique chunk
        unique_chunks = remove_duplicates(relevant_results)
        for chunk in unique_chunks:
            # Extract and parse references from the chunk
            chunk_references = []
            if chunk.get('references'):
                try:
                    reference_strings = json.loads(chunk['references'])
                    for ref_str in reference_strings:
                        parsed_ref = parse_reference(ref_str, chunk['title'], logger)
                        if parsed_ref:
                            chunk_references.append(parsed_ref)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Error decoding references in chunk '{chunk['title']}': {e}"
                    )
                    continue  # Skip this chunk or handle accordingly

            # Step 4: Assign new unique reference numbers
            old_to_new_ref_map = {}
            for ref in chunk_references:
                old_ref_number = ref.get('reference_number')
                if old_ref_number is None:
                    logger.warning(
                        f"Reference without a number in chunk '{chunk['title']}'. Skipping."
                    )
                    continue
                # Assign a new unique reference number
                new_ref_number = next_ref_number
                next_ref_number += 1
                old_to_new_ref_map[old_ref_number] = new_ref_number
                # Update the reference number in the reference item
                ref['reference_number'] = new_ref_number
                # Collect the updated reference
                all_new_references.append(ref)

            # Step 5: Update in-text reference numbers in the chunk's content
            updated_content = chunk['content']

            # Find all bracketed citations (e.g., [1], [2,3-5], etc.)
            bracket_citations = re.findall(r"\[(.*?)\]", updated_content)

            for citation in bracket_citations:
                # Extract individual citation numbers using helper functions
                citations = extract_citations(f"[{citation}]")  # Pass the entire bracket

                if not citations:
                    continue  # No valid citations found

                # Map old reference numbers to new ones
                new_citations = []
                for ref in citations:
                    try:
                        new_ref = old_to_new_ref_map.get(int(ref), ref)
                        new_citations.append(str(new_ref))
                    except ValueError:
                        logger.warning(
                            f"Invalid reference number '{ref}' in citation '[{citation}]'. Skipping."
                        )
                        continue

                if not new_citations:
                    continue  # No valid mappings found

                # Remove duplicates and sort the citations
                new_citations_sorted = sorted(
                    set(new_citations), key=lambda x: int(x)
                )

                # Reconstruct the citation string
                new_citation_str = ','.join(new_citations_sorted)

                # Replace the old citation with the new one in the content
                updated_content = updated_content.replace(
                    f"[{citation}]", f"[{new_citation_str}]"
                )

            # Update the chunk's content with the new citations
            chunk['content'] = updated_content

            # Append the updated chunk to the list
            uptodate_content_chunks.append(
                {
                    'title': chunk['title'],
                    'category': chunk.get('topic', ''),
                    'subcategory': chunk.get('subtopic', ''),
                    'content': chunk['content'],
                    'references': json.dumps(chunk_references, indent=2),  # Update references as JSON string
                }
            )

        chunk_json = json.dumps(uptodate_content_chunks)
        save_results(index, chunk_json, f"{condition_name}_uptodate_chunks", topic_dir, logger)

        # Step 7: Integrate the new content into the Outline
        outline = await refine_outline_with_uptodate(
            index=index,
            condition=condition_name,
            alternative_name=alternative_name,
            category=category,
            article=outline,
            uptodate_chunks=uptodate_content_chunks,
            model=model,
            logger=logger
        )

        # Step 8: Save the updated outline as a JSON file
        outline_json = outline.model_dump_json(indent=2)
        save_results(index, outline_json, f"{condition_name}_uptodate", topic_dir, logger)

        return outline  # Return the updated Outline object

    except Exception as e:
        logger.error(
            f"Error in integrate_uptodate_content for topic '{condition_name}': {e}",
            exc_info=True
        )
        return None


async def process_topic(
    index: int,
    conditions_df: pd.DataFrame,
    search_results: List[dict],
    model: GenerativeModel,
    semantic_client: SemanticScholarAPI,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """
    Processes a single topic query asynchronously.

    Args:
        index (int): Index of the topic.
        conditions_df (pd.DataFrame): DataFrame containing conditions.
        search_results (List[dict]): Search results for the conditions.
        model (GenerativeModel): Generative AI model.
        semantic_client (SemanticScholarAPI): Semantic Scholar API client.
        output_dir (str): Directory to save outputs.
        logger (logging.Logger): Logger for logging information.

    Returns:
        str: Final integrated article.
    """
    try:
        row = conditions_df.iloc[index]

        condition_name = row['Condition']
        alternative_name = row['Alternative Name']
        category = row['Category']
        topic = condition_name
        if alternative_name:
            topic = f"{topic} ({alternative_name})"
        
        topic_dir = os.path.join(output_dir, sanitize_filename(topic))
        os.makedirs(topic_dir, exist_ok=True)
        logger.info(f"[{index + 1}] Processing topic: {topic}")

        # Generate Outline
        outline = await generate_outline(
            index, condition_name, alternative_name, category, model, logger
        )
        logger.info(f"[{index + 1}] Generated topic: {topic}")

        outline_json = outline.model_dump_json(indent=2)
        save_results(index, outline_json, f"{topic}_outline", topic_dir, logger)

        # Integrate UpToDate Content
        article = await integrate_uptodate_content(
            index, condition_name, alternative_name, category, search_results, 
            outline, model, topic_dir, logger
        )

        # Remove and renumber bad references
        article_data = json.loads(article.model_dump_json(indent=2))
        article_data = clean_references(article_data)
        
        article_json = json.dumps(article_data, indent=2)
        save_results(index, article_json, f"{topic}_uptodate_remapped", topic_dir, logger)
        article = Article.model_validate_json(article_json)


        # Generate Search Queries and Fetch Papers
        logger.info("Calling generate_search_query_response")
        search_queries = await generate_search_query_response(
            index, condition_name, alternative_name, category, article, model, logger
        )
        parsed_queries = [q.model_dump(mode='json') for q in search_queries.root]

        json_result = json.dumps(search_queries.model_dump(), indent=2)
        save_results(index, json_result, f"{topic}_queries", topic_dir, logger)

        
        # Search Semantic Scholar
        logger.info("Calling semantic_client.query")
        semantic_results = await semantic_client.query(index, parsed_queries)
        papers = semantic_client.format_results(semantic_results)
        papers_json = json.dumps([paper.model_dump(mode='json') for paper in papers],indent=2)
        save_results(index, papers_json, f"{topic}_papers", topic_dir, logger)
        logger.info(f"[{index + 1}] Found {len(papers)} papers for: {topic}")
        
        top_papers = semantic_client.select_top_papers(papers)
        papers_json = json.dumps([paper.model_dump(mode='json') for paper in top_papers],indent=2)
        save_results(index, papers_json, f"{topic}_papers_top", topic_dir, logger)
        logger.info(f"[{index + 1}] Selected only {len(top_papers)} papers for: {topic}")

        # Filter Papers by Relevance
        logger.info(f"[{index + 1}] Filtering papers for: {topic}")
        filtered_papers = await filter_papers_by_relevancy(
          index, condition_name, alternative_name, category, top_papers, model, logger
        )
        filtered_papers_json = json.dumps([paper.model_dump(mode='json') for paper in filtered_papers],indent=2)
        save_results(index, filtered_papers_json, f"{topic}_papers_filtered", topic_dir, logger)
        logger.info(f"[{index + 1}] Filtered to {len(filtered_papers)} papers for: {topic}")

        # Integrate Papers into Article
        sourced_article = await integrate_papers(
            index, condition_name, alternative_name, category, article, 
            filtered_papers, model, logger
        )

        sourced_article_json = sourced_article.model_dump_json(indent=2)
        save_results(index, sourced_article_json, f"{topic}_sourced", topic_dir, logger)
        logger.info(f"[{index + 1}] Integrated papers for: {topic}")

        # Remove and renumber bad references
        article_data = json.loads(sourced_article_json)
        article_data = clean_references(article_data)
        
        article_json = json.dumps(article_data, indent=2)
        save_results(index, article_json, f"{topic}_sourced_remapped", topic_dir, logger)
        sourced_article = Article.model_validate_json(article_json)

        final_article = await comprehensive_edit(
            index, condition_name, alternative_name, category, sourced_article,
            model, logger
        )

        # First saving output (after comprehensive edit)
        final_article_json = final_article.model_dump_json(indent=2)
        save_results(index, final_article_json, f"{topic}_edit", topic_dir, logger)
        logger.info(
            f"[{index + 1}] Saving output from comprehensive edit: {topic}"
        )

        # Remove and renumber bad references (after comprehensive edit)
        article_data = json.loads(final_article.model_dump_json(indent=2))
        article_data = clean_references(article_data)

        # Convert article_data back to an Article object before saving
        article_json = json.dumps(article_data, indent=2)
        final_article = Article.model_validate_json(article_json)

        # Save the final article
        final_article_json = final_article.model_dump_json(indent=2)
        save_results(index, final_article_json, f"{topic}_final", topic_dir, logger)

        logger.info(f"[{index + 1}] Finished processing topic: {topic}")
        return sourced_article

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
        logger.info(f"Using model {GENERATIVE_MODEL_NAME} with temperature={TEMPERATURE}")

        # Load Conditions
        logger.info(f"Loading conditions from '{CSV_FILE}'...")
        conditions_df = pd.read_csv(CSV_FILE).fillna("")

        # Load Search Results
        logger.info(f"Loading search results from '{SEARCH_RESULTS_FILE}'...")
        try:
            with open(SEARCH_RESULTS_FILE, 'r', encoding='utf-8') as file:
                search_results = json.load(file)
            logger.info(f"Loaded results for {len(search_results)} queries.")
        except FileNotFoundError:
            logger.critical(f"Search results file '{SEARCH_RESULTS_FILE}' not found.")
            return
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON format in '{SEARCH_RESULTS_FILE}': {e}")
            return

        # Initialize API Clients and Model
        semantic_client = SemanticScholarAPI(
            api_key=SEMANTIC_SCHOLAR_API_KEY, logger=logger, 
            sjr_threshold=MIN_SJR_SCORE, min_citation_count=MIN_CITATION_COUNT, 
            )
        semantic_client.load_journal_sjr_data(JOURNALS_FILE)
        logger.info(f"Minimum Journal SJR: {MIN_SJR_SCORE}")
        logger.info(f"Minimum paper citation count: {MIN_CITATION_COUNT}")

        model = ChatGoogleGenerativeAI(model=GENERATIVE_MODEL_NAME, temperature=TEMPERATURE)

        # Concurrency Control
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        tasks = []

        for idx, row in conditions_df[110:].iterrows():
            async def bound_process(idx=idx, topic=conditions_df['Condition']):
                async with semaphore:
                    return await process_topic(
                        idx,
                        conditions_df,
                        search_results,
                        model,
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
