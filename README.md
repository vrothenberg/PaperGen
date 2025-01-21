# PaperGen: Automated Medical Knowledgebase Article Generator

[![Build Status](https://github.com/vrothenberg/PaperGen/actions/workflows/main.yml/badge.svg)](https://github.com/vrothenberg/PaperGen/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

PaperGen is a powerful tool that automates the creation of comprehensive, evidence-based medical knowledgebase articles. By intelligently integrating content from UpToDate (if available) and fetching the latest research from Semantic Scholar, PaperGen streamlines the process of producing high-quality medical content, saving researchers and writers valuable time and effort.

## Features

*   **Automated, Structured Outlines:** PaperGen uses Google Gemini to generate well-organized outlines for medical articles, ensuring consistent structure and coverage of essential topics.
*   **UpToDate Content Integration (Optional):** If you have access to UpToDate, PaperGen seamlessly integrates relevant information from your locally stored articles, enriching the content with expert knowledge.
*   **Smart Semantic Scholar Search:** Automatically finds and incorporates the most relevant scientific papers from Semantic Scholar, ensuring articles are backed by the latest research.
*   **Thorough Reference Management:** Automatically handles citations, removing duplicates, fixing bad references, and ensuring a clean, accurate bibliography.
*   **Concurrent and Efficient:** Processes multiple medical conditions simultaneously, saving time and making the most of your computational resources.
*   **Detailed Logging:** Provides comprehensive logs for easy monitoring and debugging.

## Project Structure

```
.
├── notebooks/
│   └── search_local_uptodate.ipynb      # Notebook for searching and indexing UpToDate articles
├── src/
│   ├── __init__.py
│   ├── config.py                        # Configuration and environment variables
│   ├── generators.py                    # Functions to generate and refine article outlines
│   ├── models.py                        # Pydantic models defining the article structure
│   ├── semanticscholar.py               # Semantic Scholar API integration
│   └── utils.py                         # Utility functions for retries, JSON cleaning, and saving results
├── data/
│   ├── output/                           # Directory where generated articles are saved
│   └── search_results.json               # Precomputed BM25 search index results
├── project_explorer.py                  # Script to display project structure with file contents
├── main.py                              # Script that executes the main paper generation pipeline
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (to be created)
└── README.md                            # Project documentation
```

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/vrothenberg/PaperGen.git
    cd PaperGen
    ```

2. **Create and Activate a Virtual Environment:**

    It's highly recommended to use a virtual environment to manage dependencies. You can use `conda`, `mamba`, or `venv`.

    **Using `mamba` (or `conda`):**

    ```bash
    mamba create -n papergen python=3.12
    conda activate papergen
    ```

    **Using `venv`:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Create a `.env` file in the root directory of the project with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

*   **GEMINI_API_KEY:** Your API key for Google Gemini.
*   **SEMANTIC_SCHOLAR_API_KEY:** Your API key for accessing the Semantic Scholar API.

*Note: Replace `your_gemini_api_key` and `your_semantic_scholar_api_key` with your actual API keys.*

**Important:** The `config.py` file will automatically load these environment variables using `python-dotenv`.

## Data Preparation

### UpToDate Articles (Optional)

**The pipeline can leverage locally stored UpToDate articles to enhance the generated content. If you do NOT have access to UpToDate, you can skip this section. The pipeline will still function, but without the benefit of UpToDate integration.**

1. **Obtain UpToDate Articles:**

    Ensure you have a collection of relevant UpToDate articles stored locally. These articles should be in a consistent format (e.g., JSON or Markdown) and organized in a directory structure accessible by the pipeline.

2. **Indexing Articles:**

    Use the `search_local_uptodate.ipynb` notebook to preprocess and index the UpToDate articles. This involves:

    *   Searching for relevant articles based on a list of medical conditions.
    *   Computing a BM25 search index to rank articles by relevance.
    *   Storing the top-ranked articles for each condition in `data/search_results.json`.

### Search Index

The pipeline uses a precomputed search index (using the BM25 algorithm) to quickly identify the most relevant UpToDate articles for each medical condition. The results of this indexing process are stored in `data/search_results.json`.

## Usage

To generate knowledgebase articles, run the main pipeline script:

```bash
python main.py
```

This will initiate the following process:

1. **Loading Data:** The script reads the precomputed search results (`data/search_results.json`) and the list of medical conditions to process.
2. **Generating Outlines:** For each condition, Google Gemini is used to create a detailed article outline based on a predefined Pydantic schema.
3. **Integrating UpToDate Articles (Optional):** If UpToDate integration is enabled and relevant articles are found, the outlines are enhanced with information from these articles.
4. **Generating Search Queries:** The AI generates specific search queries to find relevant scientific papers for each section of the outline.
5. **Fetching Papers:** The script uses the Semantic Scholar API to retrieve abstracts and metadata for the generated search queries.
6. **Integrating Research:** The most relevant papers are carefully integrated into the article, along with proper citations.
7. **Finalizing and Saving Articles:** The generated articles undergo a final editing pass (readability, consistency, fact-checking) and are saved as JSON files in the `data/output/` directory.

**Example Output:**

The generated JSON files in `data/output/` will have a structure similar to this (simplified):

```json
{
  "title": "Diabetes Mellitus",
  "subtitle": "A chronic metabolic disorder characterized by elevated blood sugar levels.",
  "overview": {
    "heading": "Overview",
    "content": "Diabetes mellitus is a significant global health concern..."
  },
  "references": {
    "heading": "References",
    "content": [
      {
        "reference_number": 1,
        "authors": "Smith et al.",
        "year": 2023,
        "title": "A Comprehensive Review of Diabetes Management",
        "journal_source": "Journal of Clinical Endocrinology",
        "url_doi": "https://doi.org/10.1210/jc.2023-00123"
      }
    ]
  }
}
```

## Workflow Details

1. **Preprocessing:**
    *   **Search and Indexing (Optional):** If using UpToDate, use BM25 to find and rank relevant articles for each medical condition.
    *   **Storage:** Save the top-ranked articles in `data/search_results.json`.

2. **Outline Generation:**
    *   **Prompting the Model:** Provide Google Gemini with a Pydantic schema to generate a structured outline.
    *   **Result:** A JSON-formatted outline.

3. **Integration of UpToDate Information (Optional):**
    *   **Refinement:** Enhance the outline with data from relevant UpToDate articles.
    *   **Citations:** Integrate references, ensuring professional formatting.

4. **Search Query Generation:**
    *   **Prompting the Model:** Generate search queries to find supporting scientific papers.
    *   **Result:** A list of JSON-formatted search queries.

5. **Paper Retrieval:**
    *   **Semantic Scholar API:** Execute search queries to fetch relevant papers.
    *   **Formatting:** Organize retrieved data into a JSON structure.

6. **Final Article Compilation:**
    *   **Integration:** Merge fetched papers into the article, adding citations and a references section.
    *   **Output:** Save finalized articles in `data/output/`.

## Dependencies

The project relies on the following third-party Python libraries:

*   [`python-dotenv`](https://pypi.org/project/python-dotenv/): For loading environment variables.
*   [`pydantic`](https://pydantic-docs.helpmanual.io/): For data validation and settings management using Python type annotations.
*   [`rich`](https://rich.readthedocs.io/en/stable/): For enhanced logging and terminal output.
*   [`aiohttp`](https://aiohttp.readthedocs.io/en/stable/): For asynchronous HTTP requests.
*   [`langchain-google-genai`](https://pypi.org/project/langchain-google-genai/): For interacting with Google Gemini.

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Notes

*   **UpToDate Articles (Optional):** This pipeline is designed to work with or without local UpToDate articles. If you don't have access, the generated articles will rely solely on Semantic Scholar for external information.
*   **API Keys:** Ensure your `.env` file contains valid API keys for Google Gemini and Semantic Scholar.
*   **Concurrency:** The pipeline uses asynchronous programming to handle multiple tasks efficiently.
*   **Logging:** Logs are saved in `data/output/pipeline.log` and displayed in the console.
*   **Error Handling:** The pipeline includes robust error handling with retries for network requests.
*   **Limitations:** The quality of the generated articles depends on the underlying AI model and the data it has access to. Results should be reviewed by a subject-matter expert.

## Contributing

Contributions to PaperGen are welcome! If you'd like to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and write tests if applicable.
4. Ensure your code follows the project's style guidelines (we use `black`).
5. Submit a pull request, clearly describing your changes.

## License

This project is licensed under the [MIT License](LICENSE).
