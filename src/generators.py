# src/generators.py

import json
import logging
import asyncio
from pydantic import ValidationError
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
# from langsmith import wrappers, traceable

from src.models import Article, SearchQuery, SearchQueryList, Paper

MAX_RETRIES = 10
RETRY_DELAY = 2.0


async def generate_outline_with_retries(
    index: int,
    prompt_and_model: RunnableSequence,
    parser: PydanticOutputParser,
    input_data: dict,
    logger: logging.Logger = None
) -> Article:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Invoke the model with the input data
            logger.info(f"[{index+1}] Attempt {attempt} of {MAX_RETRIES}...")
            output = await prompt_and_model.ainvoke(input_data)
            
            # Parse and validate the generated content
            outline = await parser.ainvoke(output)
            logger.info(f"[{index+1}] Success on attempt {attempt}.")
            return outline
        
        except ValidationError as e:
            logger.error(f"[{index+1}] Validation Error on attempt {attempt}: {e}")
        except Exception as ex:
            logger.error(f"[{index+1}] Error on attempt {attempt}: {ex}")

        # Wait before retrying
        if attempt < MAX_RETRIES:
            logger.info(f"[{index+1}] Retrying in {RETRY_DELAY ** attempt} seconds...")
            await asyncio.sleep(RETRY_DELAY ** attempt)
        else:
            logger.error(f"[{index+1}] Max retries reached. Aborting.")
    
    raise Exception


# @traceable(run_type="chain")
async def generate_outline(
    index: int,
    condition: str,
    alternative_name: str,
    category: str,
    model: ChatGoogleGenerativeAI,
    logger: logging.Logger = None
) -> Article:
    """
    Generates a detailed knowledgebase article outline on a given topic using a generative model.
    Focuses on content and structure, without fabricating references or links.
    """
    parser = PydanticOutputParser(pydantic_object=Article)
    # Create the prompt template with format instructions
    prompt_template = PromptTemplate(
        template="""You are a professional scientific writer tasked with developing a detailed and informative knowledgebase article outline on a given condition.

Condition: '{condition}'
Alternate Name: '{alternative_name}'
Category: '{category}'

Detailed Instructions for Each Section:

- **Title**: The main heading of the article, which is the condition itself (e.g., "Diabetes").

- **Subtitle**: A concise introductory phrase summarizing the condition.

- **Sections**: Each section in the article is structured with a specific heading and content. Ensure that all of these sections are included:

    - **Overview**
    - **Key Facts**
    - **Symptoms**
    - **Types**
    - **Causes**
    - **Risk Factors**
    - **Diagnosis**
    - **Prevention**
    - **Specialist to Visit**
    - **Treatment**
    - **Home-Care**
    - **Living With**
    - **Complications**
    - **Alternative Therapies**
    - **FAQs**
    - **References**

For each section, provide the following:

- **Heading**: The title of the section (e.g., "Overview").

- **Content**: Detailed information relevant to the section. Follow these guidelines:

    - Maintain a professional yet approachable tone.
    - **Do not include any placeholder or dummy references. The References section should be initially empty.**
    - **Do not hallucinate or fabricate any URLs or links.**
    - Where possible, include statistics, research findings, or notable insights to make the article credible and informative. These should be based on your general knowledge and not attributed to specific sources at this stage.
    - Use bullet points where specified to ensure that information is captured in the output.
    - Ensure that content is well-written and contains a brief but sufficient summary.
    - Where subtypes exist, include nested subheadings within the content using the `###` markdown header format.

    - **FAQs**: The content for the FAQs section should be a JSON array of question-answer pairs. Each pair should have a "question" field and an "answer" field. Generate 3 to 5 FAQs.

- **References**: This section should be empty. We will populate it in a later step.

---
{format_instructions}
""",
        input_variables=["condition", "alternative_name", "category"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt_template | model

    # Define the input data
    input_data = {
        "condition": condition,
        "alternative_name": alternative_name,
        "category": category
    }

    logger.info(f"[{index+1}] Generating outline...")
    # Generate the outline with retries
    outline = await generate_outline_with_retries(
        index, prompt_and_model, parser, input_data, logger
    )
    return outline


# @traceable(run_type="chain")
async def refine_outline_with_uptodate(
    index: int,
    condition: str,
    alternative_name: str,
    category: str,
    article: Article,
    uptodate_chunks: List[str],
    model: ChatGoogleGenerativeAI,
    logger: logging.Logger
) -> Article:
    """
    Refines the given outline by incorporating relevant information and citations
    from a list of UpToDate articles.
    """
    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=Article)

    # Create the prompt template with stronger emphasis on citation quality and explicit DOI instructions
    prompt_template = PromptTemplate(
        template="""You are a professional scientific writer tasked with integrating relevant information and references into an existing knowledgebase article (ARTICLE) on a given condition.

Condition: '{condition}'
Alternate Name: '{alternative_name}'
Category: '{category}'

Goal:
Enhance the ARTICLE by incorporating relevant information and insights from the provided UpToDate article snippets (UPTODATE).
Your task is to refine the ARTICLE while preserving its structure and integrity.
**Importantly, do not directly cite, refer to, or mention UpToDate articles anywhere in the output.**
Only cite the original scientific research articles that are referenced in those UpToDate articles if you choose to incorporate information from those articles.

---

Detailed Instructions:

1. Review and Analyze:
   - Examine the ARTICLE and the UPTODATE articles to identify areas where additional information, clarity, or updated data can be integrated.
   - Retain all existing content and structure unless updating with more accurate or relevant details.

2. Integrate Information:
   - Incorporate relevant data and insights from UPTODATE into appropriate sections of the ARTICLE.
   - Ensure all new information fits seamlessly within the existing framework, improving the article’s depth and accuracy.

3. Citation and Reference Style:
    - Each UPTODATE input contains a `references` field. This is a JSON encoded list of the reference strings used in the UPTODATE article. Each reference string will follow the format: `"[reference_number] citation"` where the citation field contains the full citation in APA format.
   - Use a consistent, numbered inline citation style (e.g., [1], [2], [3]) within the ARTICLE text.
   - Add a References section at the end of the ARTICLE, listing all cited works. Each reference should be a single string containing all the necessary information.
   - **Crucially, use citations judiciously and avoid over-citation.**
   - **Only include a citation when it directly supports a specific claim or provides essential context.**
   - **Do not include multiple citations for a single, general statement unless each citation offers unique and valuable information.**
   - **Aim for 1-3 citations at the end of a sentence or paragraph, only if necessary.**
   - **Strongly prefer a single, high-quality citation over multiple citations for the same point.**
   - Do not add any empty references.
   - **Prioritize non-DOI URLs when available. If a non-DOI URL is not available, and a DOI is provided, you may use it. However, do not hallucinate or fabricate DOIs or URLs that are not provided in the UPTODATE references.**
   - Do not cite directly from or mention UpToDate.

4. Preservation and Adaptation:
   - Maintain the ARTICLE’s structure and professional tone.
   - Avoid overwriting existing content unless necessary for clarity or improvement.
   - Add content to expand on existing sections or provide additional context when relevant, for example additional FAQs (3 to 10 questions).

---

Output Requirements:
- Return only the refined JSON object as specified.
- Do not include any extraneous commentary, reasoning steps, or notes outside the refined ARTICLE.
- Ensure that all integrated references are formatted in the "References" section.

---
Inputs:

ARTICLE:
{article}

UPTODATE:
{uptodate_chunks}

---
{format_instructions}
""",
        input_variables=["condition", "alternative_name", "category", "article", "uptodate_chunks"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Define input data (no changes here)
    input_data = {
        "condition": condition,
        "alternative_name": alternative_name,
        "category": category,
        "article": article.model_dump_json(indent=2),
        "uptodate_chunks": json.dumps(uptodate_chunks, indent=2)
    }

    prompt_and_model = prompt_template | model

    logger.info(f"[{index+1}] Generating article with UpToDate chunks...")
    # Generate the outline with retries
    outline = await generate_outline_with_retries(
        index, prompt_and_model, parser, input_data, logger
    )
    return outline


# @traceable(run_type="chain")
async def integrate_papers(
    index: int,
    condition_name: str,
    alternative_name: str,
    category: str,
    article: Article,
    papers: List[Paper],
    model: ChatGoogleGenerativeAI,
    logger: logging.Logger
) -> Article:
    """
    Integrate relevant scientific papers into the provided article using the model.

    Args:
        condition (str): The condition or disease topic of the article.
        alternative_name (str): The alternate name of the article, could be empty.
        category (str): The category of the condition.
        article (str): The initial article or outline to be enhanced in JSON format.
        papers (List[dict]): A list of scientific paper details to integrate.
        model (ChatGoogleGenerativeAI): The Langchain model instance for generation.
        logger (logging.Logger): The logger to use for error messages

    Returns:
        Article: The validated and revised article with integrated references in JSON format.
    """

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=Article)

    # Create the prompt template (REVISED)
    prompt_template = PromptTemplate(
        template="""You are a professional scientific writer tasked with integrating relevant references into an existing knowledgebase article (ARTICLE) on the given condition.

Condition: '{condition_name}'
Alternate Name: '{alternative_name}'
Category: '{category}'

Goal:
Refine and expand the ARTICLE by judiciously integrating the most relevant references from the provided PAPERS JSON list. These references should support the article's claims, improve its accuracy, and enhance its authority. The final output should be the revised article in JSON format, with integrated inline citations and a well-formatted reference list.

---

Input:
1. ARTICLE: An existing piece of content (provided below) discussing the condition topic in detail. This is in JSON format.
2. PAPERS: A JSON array of papers retrieved from Semantic Scholar. Each paper entry includes:
   - `section`: The ARTICLE section where the paper might be most relevant.
   - `query`: A query or topic associated with how the paper was found.
   - `title`: Title of the paper.
   - `abstract`: Abstract or summary of the paper, containing key findings and conclusions.
   - `authors`: "url": "https://www.semanticscholar.org/paper/aebc1ad4dcc5a121f477f95bc9dd02e87e804020",
   - `url`: The preferred Semantic Scholar URL for the paper. Use this if it exists.
   - `publicationVenue`: A dictionary which contains the Journal `name`.
   - `openAccessPdf`: A dictionary which contains the `url` for the open access PDF. Only use this if the preceding Semantic Scholar and Journal URLs are missing.
   - `citation`: A suggested citation, including author(s), title, year, source, DOI and/or URL.

Tasks & Guidelines:

1. Identify Claims for Enhancement:
   - Examine each section of the ARTICLE to identify claims, statistics, or statements that could benefit from additional evidence or context.
   - Use the `section` and `query` fields in the PAPERS list to locate the most relevant studies or findings.

   Subtasks:
   - Prioritize Relevance: Select claims that can be significantly improved through substantiation or deeper exploration.
   - Match Paper Insights to Claims: Analyze abstracts and conclusions to find data or findings that align with or directly support ARTICLE claims.
   - Consider Contexts: Look for papers that provide:
     - Foundational evidence (e.g., systematic reviews, meta-analyses, or key experiments).
     - Complementary findings offering alternative perspectives or added depth.
     - Emerging insights, such as novel treatments, mechanisms, or epidemiological trends.
   - Cross-Reference Sections and Queries: Papers tagged for specific sections or queries should directly inform the corresponding ARTICLE sections. However, papers with broader applicability can be integrated across multiple sections as relevant.

2. Select Relevant Papers:
    - **Crucially, be highly selective with paper inclusion. Choose only the most impactful and directly relevant papers.**
   - Use papers selectively, focusing on those that directly support or expand the claims in the ARTICLE.
   - **Strongly prefer a single, high-quality citation (e.g., a review article or meta-analysis) over multiple citations for the same point.**
   - Avoid including references solely for increasing the citation count.
   - Skip sections if no suitable paper from PAPERS aligns with their content.

3. Citation and Reference Style:
    - **Use citations sparingly and only when they add significant value.**
   - Use a consistent, numbered inline citation style (e.g., [1], [2], [3]) within the ARTICLE text.
   - **Aim for 1-2 citations at the end of a sentence or paragraph, and only if necessary to support a specific claim.**
   - Add a References section at the end of the ARTICLE, listing all cited works in APA-like formatting, including authors, publication year, title, source/journal, URL or DOI.
   - **Pay very close attention to the reference numbers. Do not use duplicate reference numbers. Each new citation must be assigned a unique, sequential number, continuing from the highest existing number in the ARTICLE.**
   - Hyperlink the URL or DOI where possible.
   - Prefer non-DOI URLs when available
   - Reformat provided citations if needed to ensure professional consistency.

4. Preserve and Integrate Gracefully:
   - Maintain the ARTICLE’s original structure, and cohesive flow.
   - Do not remove existing references or hyperlinks unless it improves accuracy or clarity.
   - Place citations inline at the end of sentences or paragraphs where they provide the strongest support.

5. Clarity and Depth:
   - Summarize and incorporate key findings or data from relevant papers to strengthen claims without adding unnecessary complexity.
   - Ensure that all added content aligns with the ARTICLE’s tone and maintains a professional, concise, and neutral style.

6. No Extraneous Commentary:
   - Return only the final revised ARTICLE in JSON format.
   - Do not include any instructions, reasoning steps, or additional commentary outside the revised ARTICLE and references.

7. Output the response as strict JSON without any additional commentary or formatting. Do not include code block markers like triple backticks (`).

---

ARTICLE:
{article}

PAPERS:
{papers}

---
{format_instructions}
""",
        input_variables=["condition_name", "alternative_name", "category", "article", "papers"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Define input data (no changes here)
    input_data = {
        "condition_name": condition_name,
        "alternative_name": alternative_name,
        "category": category,
        "article": article.model_dump_json(indent=2),
        "papers": json.dumps([paper.model_dump(mode='json') for paper in papers], indent=2)
    }

    prompt_and_model = prompt_template | model

    logger.info(f"[{index+1}] Generating article with Semantic Scholar papers...")
    # Generate the outline with retries
    article = await generate_outline_with_retries(
       index, prompt_and_model, parser, input_data, logger
    )
    return article


# @traceable(run_type="chain")
async def generate_search_queries_with_retries(
    index: int,
    prompt_and_model: RunnableSequence,
    parser: PydanticOutputParser,
    input_data: dict,
    logger: logging.Logger = None
) -> SearchQueryList:
    """
    Generates search queries with retries using a given prompt and model.

    Args:
        prompt_and_model: A LangChain Runnable representing the prompt and model.
        parser: A PydanticOutputParser for parsing the output into SearchQuery objects.
        input_data: The input data for the prompt, including the outline.
        logger: Logger instance.

    Returns:
        A list of SearchQuery objects.

    Raises:
        Exception: If the maximum number of retries is reached without success.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Invoke the model with the input data
            logger.info(f"[{index+1}] Search query generation attempt {attempt} of {MAX_RETRIES}...")
            output = await prompt_and_model.ainvoke(input_data)

            # Parse and validate the generated content
            search_queries = await parser.ainvoke(output)
            logger.info(f"[{index+1}] Search query generation success on attempt {attempt}.")
            return search_queries

        except ValidationError as e:
            logger.error(f"[{index+1}] Search query validation error on attempt {attempt}: {e}")
        except Exception as ex:
            logger.error(f"[{index+1}] Search query generation error on attempt {attempt}: {ex}")

        # Wait before retrying
        if attempt < MAX_RETRIES:
            logger.info(f"[{index+1}] Retrying search query generation in {RETRY_DELAY  ** attempt} seconds...")
            await asyncio.sleep(RETRY_DELAY ** attempt)
        else:
            logger.error(f"[{index+1}] Max retries reached for search query generation. Aborting.")

    raise Exception(f"Failed to generate search queries after {MAX_RETRIES} retries.")


# @traceable(run_type="chain")
async def generate_search_query_response(
    index: int,
    condition_name: str,
    alternative_name: str, 
    category: str, 
    article: Article,
    model: ChatGoogleGenerativeAI,
    logger: logging.Logger = None
) -> SearchQueryList:
    """
    Generates search queries for each section of a given outline using a language model.

    Args:
        index (int)
        article (Outline): The article outline.
        model (ChatGoogleGenerativeAI): The language model.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        SearchQueryList: A list of search queries, each corresponding to a section of the outline.
    """
    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=SearchQueryList)

    # Create the prompt template
    prompt_template = PromptTemplate(
        template="""You are tasked with generating search queries to find corroborating evidence for key claims in a knowledgebase article.
The goal is to identify relevant scientific papers to support and enhance the ARTICLE, ensuring credibility and depth.

Condition: '{condition}'
Alternate Name: '{alternative_name}'
Category: '{category}'

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


ARTICLE:
{article}

---
{format_instructions}
        """,
        input_variables=["condition", "alternative_name", "category", "article"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Define input data
    input_data = {
        "condition": condition_name,
        "alternative_name": alternative_name,
        "category": category,
        "article": article.model_dump_json(indent=2),
    }

    prompt_and_model = prompt_template | model

    # Generate the search queries with retries
    search_queries = await generate_search_queries_with_retries(
        index, prompt_and_model, parser, input_data, logger
    )
    return search_queries


# @traceable(run_type="chain")
async def comprehensive_edit(
    index: int,
    condition_name: str,
    alternative_name: str,
    category: str,
    article: Article,
    model: ChatGoogleGenerativeAI,
    logger: logging.Logger
) -> Article:
    """
    Performs a comprehensive edit of the article, combining readability improvements,
    tone and style consistency checks, fact-checking, and reference consolidation.

    Args:
        index: Index of the article in the processing pipeline.
        condition_name: Name of the condition.
        alternative_name: Alternative name of the condition.
        category: Category of the condition.
        article: The article to be edited.
        model: The LLM for processing.
        logger: Logger instance.

    Returns:
        The comprehensively edited article.
    """
    parser = PydanticOutputParser(pydantic_object=Article)

    prompt_template = PromptTemplate(
        template="""You are a highly skilled and detail-oriented scientific editor tasked with performing a comprehensive edit of a knowledgebase article.

Condition: '{condition_name}'
Alternate Name: '{alternative_name}'
Category: '{category}'

YOUR TASK:
Perform the following edits on the provided ARTICLE, ensuring the highest standards of quality, accuracy, and consistency.

1. **Readability and Clarity**:
   - Rephrase sentences or paragraphs that are complex, ambiguous, or unclear to improve overall readability.
   - Enhance conciseness by eliminating redundancy, wordiness, and unnecessary jargon.
   - Ensure smooth transitions between sentences and paragraphs for a coherent flow.

2. **Tone, Style, and Formatting Consistency**:
   - Maintain a formal, objective, and scientific tone throughout the article. Adjust any sections that deviate from this tone.
   - Ensure uniformity in writing style, including the use of terminology, abbreviations, and acronyms. Standardize variations where necessary.
   - Check for consistency in formatting elements such as headings, subheadings, bullet points, and inline citations. Make adjustments to maintain a uniform format.

3. **Fact-Checking and Corrections**:
   - Identify statements, data, statistics, and assertions that can be fact-checked.
   - Verify the accuracy of each identified claim using your extensive knowledge base.
   - Correct any inaccuracies or outdated information, preserving the original meaning and intent as much as possible.
   - If a correction significantly alters the original information, and you have access to a reliable source to support the correction, briefly note the source within the text (e.g., "[Source: Author, Year]") without adding it to the existing references.
   - Briefly note any significant changes or corrections made within the text itself (e.g., "[Corrected: Previous statement was updated based on recent research.]"). This should be done sparingly and only when necessary for transparency.

4. **Reference Consolidation**:
    - Identify any references in the "References" section that have the same reference number but different citation details. These are considered duplicates.
    - Merge: For each set of duplicate references, choose the most comprehensive and accurate citation as the primary reference.
    - Remap: Assign a new, unique reference number to the primary reference. Remove the other duplicate entries.
    - Update Inline Citations: Throughout the article's content, replace all instances of the old duplicate reference numbers with the new, unique reference number assigned to the primary reference.
    - Preserve Order: Maintain the relative order of references based on their first appearance in the text. Renumber the references sequentially after merging.

5. **Citation Style**:
    - Ensure that inline citations and the "References" section follow a consistent style.
    - Correct any inconsistencies in the formatting of existing references.

6. **Preservation**:
   - Maintain the original structure of the article, including the order of sections, headings, and subheadings.
   - Do not alter the factual content or meaning of the information presented, except when correcting inaccuracies.
   - Keep the existing references intact unless consolidating duplicates.

7. **Output**:
   - Ensure that the output is the corrected article in the original JSON format.
   - Do not include any extraneous commentary, reasoning steps, or notes outside the refined ARTICLE.
   - Output the response as strict JSON without any additional commentary or formatting. Do not include code block markers like triple backticks (`).

ARTICLE:
{article}

---
{format_instructions}
""",
        input_variables=["condition_name", "alternative_name", "category", "article"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    input_data = {
        "condition_name": condition_name,
        "alternative_name": alternative_name,
        "category": category,
        "article": article.model_dump_json(indent=2),
    }

    prompt_and_model = prompt_template | model

    logger.info(f"[{index+1}] Performing comprehensive edit of the article...")
    edited_article = await generate_outline_with_retries(
        index, prompt_and_model, parser, input_data, logger
    )

    return edited_article