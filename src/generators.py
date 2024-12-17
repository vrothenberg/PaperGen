
import json
import random
import asyncio
import logging
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field, validator, ValidationError
from vertexai.generative_models import GenerativeModel

from src.utils import retry_with_backoff, clean_json


class ReferenceItem(BaseModel):
    reference_number: int = Field(..., description="Sequential reference number.")
    authors: str = Field(..., description="Author(s) of the reference.")
    year: str = Field(..., description="Publication year.")
    title: str = Field(..., description="Title of the reference.")
    journal_source: str = Field(..., description="Journal or source of publication.")
    url_doi: str = Field(..., description="URL or DOI of the reference.")

class FAQItem(BaseModel):
    question: str = Field(..., description="A frequently asked question about the condition.")
    answer: str = Field(..., description="A concise and informative answer to the question.")

class OverviewSection(BaseModel):
    heading: str = Field("Overview", const=True)
    content: str = Field(..., description="A high-level introduction to the topic, explaining its significance and impact, key statistics, and facts.")

class KeyFactsSection(BaseModel):
    heading: str = Field("Key Facts", const=True)
    content: List[str] = Field(..., description="Notable statistics or key data, presented as a list of strings.")

class SymptomsSection(BaseModel):
    heading: str = Field("Symptoms", const=True)
    content: List[str] = Field(..., description="Common signs and symptoms, presented as a list of strings.")

class TypesSection(BaseModel):
    heading: str = Field("Types", const=True)
    content: str = Field(..., description="Classifications or subtypes, each with a brief explanation. Types and subtypes may use nested subheadings (###, ####).")

class CausesSection(BaseModel):
    heading: str = Field("Causes", const=True)
    content: str = Field(..., description="Explanation of underlying mechanisms or causes, including primary causes.")

class RiskFactorsSection(BaseModel):
    heading: str = Field("Risk Factors", const=True)
    content: List[str] = Field(..., description="Factors that increase susceptibility, including lifestyle, genetic, or environmental risk factors (intended for bullet points).")

class DiagnosisSection(BaseModel):
    heading: str = Field("Diagnosis", const=True)
    content: str = Field(..., description="Outline of the diagnostic process, including medical history, clinical symptoms, tests, or imaging tools. Can include subheadings (###) for specific methods.")

class PreventionSection(BaseModel):
    heading: str = Field("Prevention", const=True)
    content: List[str] = Field(..., description="Practical advice for risk reduction, with evidence-based recommendations (intended for bullet points or numbered lists).")

class SpecialistToVisitSection(BaseModel):
    heading: str = Field("Specialist to Visit", const=True)
    content: str = Field(..., description="Healthcare providers involved in diagnosis and treatment, and their roles.")

class TreatmentSection(BaseModel):
    heading: str = Field("Treatment", const=True)
    content: str = Field(..., description="Description of medical and therapeutic interventions, including conventional, advanced, or emerging therapies.")

class HomeCareSection(BaseModel):
    heading: str = Field("Home-Care", const=True)
    content: List[str] = Field(..., description="Tips for self-management, such as lifestyle adjustments, routines, or home remedies (intended for bullet points).")

class LivingWithSection(BaseModel):
    heading: str = Field("Living With", const=True)
    content: str = Field(..., description="Guidance for long-term management, including emotional, social, or physical adaptation strategies.")

class ComplicationsSection(BaseModel):
    heading: str = Field("Complications", const=True)
    content: str = Field(..., description="Discussion of potential health challenges if the condition is untreated or poorly managed.")

class AlternativeTherapiesSection(BaseModel):
    heading: str = Field("Alternative Therapies", const=True)
    content: str = Field(..., description="Summary of non-conventional approaches (e.g., holistic care, acupuncture, supplements, etc.). Emphasize consulting a healthcare provider before use.")

class FAQsSection(BaseModel):
    heading: str = Field("FAQs", const=True)
    content: List[FAQItem] = Field(..., description="A list of frequently asked questions and their corresponding answers.")

class ReferencesSection(BaseModel):
    heading: str = Field("References", const=True)
    content: List[ReferenceItem] = Field(..., description="A list of references in APA-like style.")

class Outline(BaseModel):
    title: str = Field(..., description="The main heading of the article, which is the condition itself (e.g., 'Diabetes').")
    subtitle: str = Field(..., description="A concise introductory phrase summarizing the condition.")
    sections: List[Union[
        OverviewSection, KeyFactsSection, SymptomsSection, TypesSection,
        CausesSection, RiskFactorsSection, DiagnosisSection, PreventionSection,
        SpecialistToVisitSection, TreatmentSection, HomeCareSection,
        LivingWithSection, ComplicationsSection, AlternativeTherapiesSection,
        FAQsSection, ReferencesSection
    ]] = Field(..., description="A list of sections in the article, each with a heading and content.")



async def generate_outline(topic: str, model: GenerativeModel, logger: logging.Logger = None) -> str:
    """
    Generate a detailed knowledgebase article outline on a given topic using a generative model.
    The outline follows a predefined structure and style requirements.
    """
    prompt = f"""You are a professional scientific writer tasked with developing a detailed and informative knowledgebase article outline on the condition: '{topic}'.

---

The output must be a JSON object with the following structure:

{{
  "title": "Title of the Condition",
  "subtitle": "A brief introductory subheading providing an overview of the condition.",
  "sections": [
      {{
          "heading": "Overview",
          "content": "high-level introduction to the topic, explaining its significance and impact, key statistics, and facts." // A single string
      }},
      {{
          "heading": "Key Facts",
          "content": ["Notable statistics and key data", ...] // List of strings
      }},
      {{
        "heading": "Symptoms",
        "content": ["Common signs and symptoms in bullet points.", ...] // List of strings
      }},
     {{
          "heading": "Types",
           "content": "Classifications or subtypes, each with a brief explanation. Subtypes may use nested subheadings (###)."
      }},
      {{
          "heading": "Causes",
          "content": "Explanation of underlying mechanisms or causes, including primary causes."
      }},
      {{
          "heading": "Risk Factors",
          "content": "Factors that increase susceptibility, including lifestyle, genetic, or environmental risk factors (bullet points)."
      }},
        {{
          "heading": "Diagnosis",
          "content": "Outline of the diagnostic process, including medical history, clinical symptoms, tests, or imaging tools. Can include subheadings (###) for specific methods."
      }},
      {{
          "heading": "Prevention",
          "content": "Practical advice for risk reduction, with evidence-based recommendations (bullet points or numbered lists)."
       }},
       {{
            "heading": "Specialist to Visit",
            "content": "Healthcare providers involved in diagnosis and treatment, and their roles."
        }},
       {{
            "heading":"Treatment",
           "content":"Description of medical and therapeutic interventions, including conventional, advanced, or emerging therapies."
        }},
        {{
            "heading": "Home-Care",
            "content": "Tips for self-management, such as lifestyle adjustments, routines, or home remedies (bullet points)."
        }},
         {{
            "heading": "Living With",
            "content":"Guidance for long-term management, including emotional, social, or physical adaptation strategies."
        }},
        {{
            "heading":"Complications",
            "content": "Discussion of potential health challenges if the condition is untreated or poorly managed."
        }},
       {{
          "heading": "Alternative Therapies",
          "content": "Summary of non-conventional approaches (e.g. holistic care, acupuncture, supplements, etc.). Emphasize consulting a healthcare provider before use."
       }},
        {{
          "heading": "FAQs",
          "content": [
            {{
              "question": "A frequently asked question about the condition.",
              "answer": "A concise and informative answer to the question."
            }},
            {{
              "question": "Another common question regarding the condition.",
               "answer": "A clear and helpful response to the question."
            }},
            ...,
            {{
              "question": "Last slightly less frequently asked question.",
               "answer": "Reliable and informational answer to the FAQ."
            }}
           ]
        }},
        {{
      "heading": "References",
      "content": [
          {{
            "reference_number": 1,
            "authors": "Author(s)",
            "year": "Year",
            "title": "Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }},
          {{
           "reference_number": 2,
            "authors": "Another Author(s)",
            "year": "Year",
            "title": "Another Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }},
          ...,
          {{
           "reference_number": N,
            "authors": "Author(s)",
            "year": "Year",
            "title": "Last Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }},

        ]
      }}
    ]
}}

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

Structure:
The ARTICLE should follow this structure, it is already in JSON format:

{{
  "title": "Title of the Condition",
  "subtitle": "A brief introductory subheading providing an overview of the condition.",
  "sections": [
      {{
          "heading": "Overview",
          "content": "A high-level introduction to the topic, explaining its significance and impact, key statistics, and facts."
      }},
      {{
          "heading": "Key Facts",
          "content": "Notable statistics or key data, presented in bullet points or a table."
      }},
      {{
        "heading": "Symptoms",
        "content": "A list of common signs and symptoms in bullet points."
      }},
     {{
          "heading": "Types",
           "content": "Classifications or subtypes, each with a brief explanation. Subtypes may use nested subheadings (###)."
      }},
      {{
          "heading": "Causes",
          "content": "Explanation of underlying mechanisms or causes, including primary causes."
      }},
      {{
          "heading": "Risk Factors",
          "content": "Factors that increase susceptibility, including lifestyle, genetic, or environmental risk factors (bullet points)."
      }},
        {{
          "heading": "Diagnosis",
          "content": "Outline of the diagnostic process, including medical history, clinical symptoms, tests, or imaging tools. Can include subheadings (###) for specific methods."
      }},
      {{
          "heading": "Prevention",
          "content": "Practical advice for risk reduction, with evidence-based recommendations (bullet points or numbered lists)."
       }},
       {{
            "heading": "Specialist to Visit",
            "content": "Healthcare providers involved in diagnosis and treatment, and their roles."
        }},
       {{
            "heading":"Treatment",
           "content":"Description of medical and therapeutic interventions, including conventional, advanced, or emerging therapies."
        }},
        {{
            "heading": "Home-Care",
            "content": "Tips for self-management, such as lifestyle adjustments, routines, or home remedies (bullet points)."
        }},
         {{
            "heading": "Living With",
            "content":"Guidance for long-term management, including emotional, social, or physical adaptation strategies."
        }},
        {{
            "heading":"Complications",
            "content": "Discussion of potential health challenges if the condition is untreated or poorly managed."
        }},
       {{
          "heading": "Alternative Therapies",
          "content": "Summary of non-conventional approaches (e.g., acupuncture, supplements). Emphasize consulting a healthcare provider before use."
       }},
        {{
          "heading": "FAQs",
          "content": [
            {{
              "question": "A frequently asked question about the condition.",
              "answer": "A concise and informative answer to the question."
            }},
            ...
            {{
              "question": "Other common question regarding the condition.",
               "answer": "A clear and helpful response to the question."
            }}
           ]
        }},
        {{
      "heading": "References",
      "content": [
          {{
            "reference_number": 1,
            "authors": "Author(s)",
            "year": "Year",
            "title": "Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }},
          {{
           "reference_number": 2,
            "authors": "Another Author(s)",
            "year": "Year",
            "title": "Another Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }}
        ]
      }}
    ]
}}


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

Structure:
The ARTICLE should follow this structure, it is already in JSON format:

{{
  "title": "Title of the Condition",
  "subtitle": "A brief introductory subheading providing an overview of the condition.",
  "sections": [
      {{
          "heading": "Overview",
          "content": "A high-level introduction to the topic, explaining its significance and impact, key statistics, and facts."
      }},
      {{
          "heading": "Key Facts",
          "content": "Notable statistics or key data, presented in bullet points or a table."
      }},
      {{
        "heading": "Symptoms",
        "content": "A list of common signs and symptoms in bullet points."
      }},
     {{
          "heading": "Types",
           "content": "Classifications or subtypes, each with a brief explanation. Subtypes may use nested subheadings (###)."
      }},
      {{
          "heading": "Causes",
          "content": "Explanation of underlying mechanisms or causes, including primary causes."
      }},
      {{
          "heading": "Risk Factors",
          "content": "Factors that increase susceptibility, including lifestyle, genetic, or environmental risk factors (bullet points)."
      }},
        {{
          "heading": "Diagnosis",
          "content": "Outline of the diagnostic process, including medical history, clinical symptoms, tests, or imaging tools. Can include subheadings (###) for specific methods."
      }},
      {{
          "heading": "Prevention",
          "content": "Practical advice for risk reduction, with evidence-based recommendations (bullet points or numbered lists)."
       }},
       {{
            "heading": "Specialist to Visit",
            "content": "Healthcare providers involved in diagnosis and treatment, and their roles."
        }},
       {{
            "heading":"Treatment",
           "content":"Description of medical and therapeutic interventions, including conventional, advanced, or emerging therapies."
        }},
        {{
            "heading": "Home-Care",
            "content": "Tips for self-management, such as lifestyle adjustments, routines, or home remedies (bullet points)."
        }},
         {{
            "heading": "Living With",
            "content":"Guidance for long-term management, including emotional, social, or physical adaptation strategies."
        }},
        {{
            "heading":"Complications",
            "content": "Discussion of potential health challenges if the condition is untreated or poorly managed."
        }},
       {{
          "heading": "Alternative Therapies",
          "content": "Summary of non-conventional approaches (e.g., acupuncture, supplements). Emphasize consulting a healthcare provider before use."
       }},
        {{
          "heading": "FAQs",
          "content": [
            {{
              "question": "A frequently asked question about the condition.",
              "answer": "A concise and informative answer to the question."
            }},
            ...
            {{
              "question": "Other common question regarding the condition.",
               "answer": "A clear and helpful response to the question."
            }}
           ]
        }},
        {{
      "heading": "References",
      "content": [
          {{
            "reference_number": 1,
            "authors": "Author(s)",
            "year": "Year",
            "title": "Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }},
          {{
           "reference_number": 2,
            "authors": "Another Author(s)",
            "year": "Year",
            "title": "Another Title",
            "journal_source": "Journal/Source",
             "url_doi": "URL/DOI"
          }}
        ]
      }}
    ]
}}

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