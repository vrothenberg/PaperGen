
import json
import logging
from typing import List
from google.generativeai import GenerativeModel

from src.utils import retry_with_backoff, clean_json, parse_search_queries
from src.models import Outline, SearchQuery

async def generate_outline(topic: str, model: GenerativeModel, logger: logging.Logger = None) -> str:
    """
    Generate a detailed knowledgebase article outline on a given topic using a generative model.
    The outline follows a predefined structure and style requirements.
    """
    prompt = f"""You are a professional scientific writer tasked with developing a detailed and informative knowledgebase article outline on the condition: '{topic}'.

---
The output MUST be a JSON object conforming to the following Pydantic model structure:

```json
{json.dumps(Outline.model_json_schema(), indent=2)}
```

---

Detailed Instructions for Each Section:

- **Title**: The main heading of the article, which is the condition itself (e.g., "Diabetes").

- **Subtitle**: A concise introductory phrase summarizing the condition.

- **Sections**: Each section in the article is structured as a key-value pair inside an array. Each element in the array must contain a `"heading"` and `"content"` entry.

    - **Heading**: A string providing the title of the heading. Ensure that all of these headings are included: `"Overview"`, `"Key Facts"`, `"Symptoms"`, `"Types"`, `"Causes"`, `"Risk Factors"`, `"Diagnosis"`, `"Prevention"`, `"Specialist to Visit"`, `"Treatment"`, `"Home-Care"`, `"Living With"`, `"Complications"`, `"Alternative Therapies"`, `"FAQs"`, `"References"`.

    - **Content**: The content for the associated heading. You must adhere to the following requirements:

        - Maintain a professional yet approachable tone.
        - Include hypertext links to relevant sources where appropriate, and format them in the "References" section, in the same way as a traditional scientific paper.
        - Where possible, include statistics, research findings, or notable insights to make the article credible and informative.
        - Use bullet points where specified to ensure that information is captured in the output.
        - Ensure that content is well-written and contains a brief but sufficient summary.
        - Where subtypes exist, include nested subheadings within the `content` using the `###` markdown header format.

        - **FAQs**: The `content` for the `FAQs` section should be a JSON array of question-answer pairs. Each pair should have a "question" field and an "answer" field. The questions and answers should be concise, clear, and relevant to the condition. Generate 3 to 5 FAQs.

---

Output Requirements:

- Return only the refined JSON object as specified above.
- Do not add any extraneous commentary, reasoning steps, or notes outside the JSON object.
- Ensure that all integrated references are formatted in an APA-like style and included in the `"References"` section.
"""
    try:
        async def call_generate_content():
            return (await model.generate_content_async(prompt)).text

        response = await retry_with_backoff(call_generate_content, logger=logger)
        return clean_json(response)
    except Exception as e:
        logger.error(f"Error in integrate_papers for topic {topic}: {e}", exc_info=True)
        return None


async def refine_outline_with_uptodate(
    topic: str,
    outline: str,
    uptodate_articles: List[str],
    model: GenerativeModel,
    logger: logging.Logger
) -> str:
    """
    Refines the given outline by incorporating relevant information and citations
    from a list of UpToDate articles.
    """
    prompt = f"""You are a professional scientific writer tasked with integrating relevant references into an existing knowledgebase article (ARTICLE) on the condition: '{topic}'.

Goal:
Enhance the ARTICLE by incorporating additional information and insights from the provided UpToDate articles (UPTODATE). Your task is to refine the ARTICLE while preserving its structure and integrity. **Importantly, do not directly cite, refer to, or mention UpToDate articles anywhere in the output.** Only cite the original scientific research articles that are referenced in those UpToDate articles if you choose to incorporate information from those articles, but prefer to use other URLs than the https://doi.org links, if possible.

---

The output MUST be a JSON object conforming to the following Pydantic model structure:

```json
{json.dumps(Outline.model_json_schema(), indent=2)}
```

---

Task:

1. Review and Analyze:
   - Examine the ARTICLE and the UPTODATE articles to identify areas where additional information, clarity, or updated data can be integrated.
   - Retain all existing content and structure unless updating with more accurate or relevant details.

2. Integrate Information:
   - Incorporate relevant data and insights from UPTODATE into appropriate sections of the ARTICLE.
   - Ensure all new information fits seamlessly within the existing framework, improving the article’s depth and accuracy.

3. Citation and Reference Style:
   - Use a consistent, numbered inline citation style (e.g., [1], [2], [3]) within the ARTICLE text.
   - Add a References section at the end of the ARTICLE, listing all cited works in APA-like formatting, including authors, publication year, title, source/journal, URL or DOI.
   - Hyperlink the URL or DOI where possible.
   - **Prioritize non-DOI URLs when available**.
   - **Do not cite directly from or mention UpToDate. If you use information from UpToDate, cite the original research articles it references instead.**
   - Reformat provided citations if needed to ensure professional consistency.

4. Preservation and Adaptation:
   - Maintain the ARTICLE’s structure and professional tone.
   - Avoid overwriting existing content unless necessary for clarity or improvement.
   - Add content to expand on existing sections or provide additional context when relevant, for example additional FAQs (3 to 10 questions).

5. Relevance and Redundancy:
   - Integrate only information that directly supports or enhances claims in the ARTICLE.
   - Do not repeat content already present unless it is significantly rephrased to add clarity or value.

6. Formatting and Tone:
   - Maintain a concise, professional tone.

---

Output Requirements:
- Return only the refined JSON object, which is the refined version of the ARTICLE. Do not include code block markers like triple backticks (`).
- Include all integrated references properly formatted and placed in the "References" section.
- Do not add any extraneous commentary, reasoning steps, or notes outside the refined ARTICLE.
- The output must be valid JSON. Output the response as strict JSON without any additional commentary or formatting. 

---

Inputs:

ARTICLE:
{outline}

UPTODATE:
{uptodate_articles}
"""

    try:
        async def call_generate_content():
            return (await model.generate_content_async(prompt)).text

        response = await retry_with_backoff(call_generate_content, logger=logger)
        return clean_json(response)
    except Exception as e:
        logger.error(f"Error in integrate_papers for topic {topic}: {e}", exc_info=True)
        return None


async def integrate_papers(topic:str, article: str, papers: List[dict], model: GenerativeModel, logger: logging.Logger) -> str:
    """
    Integrate relevant scientific papers into the provided article using the model.

    Args:
        topic (str): The topic of the article.
        article (str): The initial article or outline to be enhanced in JSON format.
        papers (List[dict]): A list of scientific paper details to integrate.
        model (GenerativeModel): The generative model instance for processing.
        logger (logging.Logger): The logger to use for error messages

    Returns:
        str: The revised and complete article with integrated references in JSON format.
    """

    prompt = f"""You are a professional scientific writer tasked with integrating relevant references into an existing knowledgebase article (ARTICLE) on the topic: '{topic}'.

Goal:
Refine and expand the ARTICLE by judiciously integrating references from the provided PAPERS JSON list. These references should support the article's claims, improve its accuracy, and enhance its authority. The final output should be the revised article in JSON format, with integrated inline citations and a well-formatted reference list.

---

The output MUST be a JSON object conforming to the following Pydantic model structure:

```json
{json.dumps(Outline.model_json_schema(), indent=2)}
```

---

Input:
1. ARTICLE: An existing piece of content (provided below) discussing the condition topic in detail. This is in JSON format.
2. PAPERS: A JSON array of papers retrieved from PubMed and/or Semantic Scholar. Each paper entry includes:
   - `section`: The ARTICLE section where the paper might be most relevant.
   - `query`: A query or topic associated with how the paper was found.
   - `title`: Title of the paper.
   - `abstract`: Abstract or summary of the paper, containing key findings and conclusions.
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
   - Use papers selectively, focusing on those that directly support or expand the claims in the ARTICLE.
   - Avoid including references solely for increasing the citation count.
   - Skip sections if no suitable paper from PAPERS aligns with their content.

3. Citation and Reference Style:
   - Use a consistent, numbered inline citation style (e.g., [1], [2], [3]) within the ARTICLE text.
   - Add a References section at the end of the ARTICLE, listing all cited works in APA-like formatting, including authors, publication year, title, source/journal, URL or DOI.
   - Hyperlink the URL or DOI where possible.
   - **Prefer non-DOI URLs when available**
   - Reformat provided citations if needed to ensure professional consistency.

4. Preserve and Integrate Gracefully:
   - Maintain the ARTICLE’s original structure, and cohesive flow.
   - Do not remove existing references or hyperlinks unless it improves accuracy or clarity.
   - Place citations inline at appropriate points in the text to enhance the readability and authority of the ARTICLE.

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
{json.dumps(papers, indent=2)}
"""
    try:
        async def call_generate_content():
            return (await model.generate_content_async(prompt)).text

        response = await retry_with_backoff(call_generate_content, logger=logger)
        return clean_json(response)
    except Exception as e:
        logger.error(f"Error in integrate_papers for topic {topic}: {e}", exc_info=True)
        return None
    


async def generate_search_query_response(outline: str, model: GenerativeModel, logger: logging.Logger = None) -> List[SearchQuery]:
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
    - Output the response as strict valid JSON without any additional commentary or formatting. Do not include code block markers like triple backticks (`).
    """
    try:
        async def call_generate_content():
            response = await model.generate_content_async(prompt)
            response_text = clean_json(response.text)  # Extract the response text
            return parse_search_queries(response_text)  # Parse after generation
    
        return await retry_with_backoff(call_generate_content, logger=logger)
    
    except Exception as e:
        logger.error(f"Error in generate_search_query_response for topic: {e}", exc_info=True)
        return None

