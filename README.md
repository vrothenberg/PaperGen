# PaperGen

**PaperGen** is an automated pipeline designed to generate comprehensive knowledgebase articles on various medical conditions. Leveraging advanced generative AI models, it integrates information from locally stored UpToDate articles and retrieves relevant scientific papers from PubMed and Semantic Scholar to produce well-structured, informative, and evidence-based content.

## Table of Contents

- [PaperGen](#papergen)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Data Preparation](#data-preparation)
    - [UpToDate Articles](#uptodate-articles)
    - [Search Index](#search-index)
  - [Usage](#usage)
  - [Workflow Overview](#workflow-overview)
  - [Dependencies](#dependencies)
  - [Notes](#notes)
  - [License](#license)

## Features

- **Automated Outline Generation**: Utilizes generative AI models to create structured outlines based on predefined Pydantic class structures.
- **Integration of UpToDate Articles**: Incorporates relevant information from locally stored UpToDate articles to enhance article depth and accuracy.
- **Scientific Paper Retrieval**: Searches PubMed and Semantic Scholar APIs to find and integrate relevant scientific papers supporting the article's content.
- **Concurrent Processing**: Efficiently handles multiple topics simultaneously using asynchronous programming and concurrency control.
- **Robust Logging**: Implements comprehensive logging to track the pipeline's progress and troubleshoot issues effectively.

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
│   ├── pubmed.py                        # PubMed API integration
│   ├── semanticscholar.py               # Semantic Scholar API integration
│   └── utils.py                         # Utility functions for retries, JSON cleaning, and saving results
├── data/
│   ├── output/                           # Directory where generated articles are saved
│   └── search_results.json               # Precomputed BM25 search index results
├── project_explorer.py                  # Script to display project structure with file contents
├── main.py                              # Main pipeline script
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (to be created)
└── README.md                            # Project documentation
```

## Installation

Follow the steps below to set up the PaperGen environment:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vrothenberg/PaperGen.git
   cd PaperGen
   ```

2. **Create a Virtual Environment**

   It's recommended to use `mamba` or `conda` for environment management.

   ```bash
   mamba create -n papergen python=3.12
   conda activate papergen
   ```

   *Alternatively, using `venv`:*

   ```bash
   python3 -m venv papergen_env
   source papergen_env/bin/activate
   ```

3. **Install Dependencies**

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not present, install dependencies manually:*

   ```bash
   pip install python-dotenv
   pip install pydantic
   pip install rich
   pip install aiohttp
   pip install google-generativeai
   ```

## Configuration

Create a `.env` file in the root directory of the project with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

- **GEMINI_API_KEY**: Your API key for the generative AI model (e.g., Gemini).
- **SEMANTIC_SCHOLAR_API_KEY**: Your API key for accessing the Semantic Scholar API.

*Note: Replace `your_gemini_api_key` and `your_semantic_scholar_api_key` with your actual API keys.*

## Data Preparation

### UpToDate Articles

The pipeline relies on locally stored UpToDate articles to enhance the generated knowledgebase articles. To prepare the necessary data:

1. **Obtain UpToDate Articles**

   Ensure you have a collection of relevant UpToDate articles stored locally. These articles should be in a consistent format (e.g., JSON or Markdown) and organized in a directory structure accessible by the pipeline.

2. **Indexing Articles**

   Use the `search_local_uptodate.ipynb` notebook to preprocess and index the UpToDate articles. This involves:

   - Searching for relevant articles based on a list of medical conditions.
   - Computing a BM25 search index to rank articles by relevance.
   - Storing the top-ranked articles for each condition in `data/search_results.json`.

   *Note: Other users need to adapt their implementation to have these files or skip this step if they don't possess local UpToDate articles.*

### Search Index

The search index is precomputed using the BM25 algorithm to identify the most relevant UpToDate articles for each condition. The results are stored in `data/search_results.json`.

## Usage

Run the main pipeline script to generate knowledgebase articles:

```bash
python main.py
```

This will initiate the following process:

1. **Loading Data**: Reads the precomputed search results from `data/search_results.json`.
2. **Generating Outlines**: Uses the generative model to create article outlines based on the Pydantic class structure.
3. **Integrating UpToDate Articles**: Enhances the outlines with information from the top-ranked UpToDate articles.
4. **Generating Search Queries**: Produces search queries to find relevant scientific papers.
5. **Fetching Papers**: Retrieves abstracts and metadata from PubMed and Semantic Scholar APIs.
6. **Finalizing Articles**: Integrates the fetched papers into the articles to produce the final version.
7. **Saving Results**: Outputs the generated articles in JSON format to the `data/output/` directory.

## Workflow Overview

1. **Preprocessing**

   - **Search and Indexing**: Utilize BM25 to find and rank UpToDate articles relevant to each medical condition.
   - **Storage**: Save the top-ranked articles in `data/search_results.json`.

2. **Outline Generation**

   - **Prompting the Model**: Provide the generative AI model with a Pydantic schema to generate a structured outline for the knowledgebase article.
   - **Result**: Obtain a JSON-formatted outline adhering to the predefined structure.

3. **Integration of UpToDate Information**

   - **Refinement**: Enhance the generated outline by incorporating data from relevant UpToDate articles.
   - **Citations**: Integrate references without directly mentioning UpToDate, ensuring citations are formatted professionally.

4. **Search Query Generation**

   - **Prompting the Model**: Generate targeted search queries based on the refined outline to find supporting scientific papers.
   - **Result**: Receive a list of JSON-formatted search queries.

5. **Paper Retrieval**

   - **PubMed and Semantic Scholar APIs**: Execute the search queries to fetch relevant papers, including abstracts and citations.
   - **Formatting**: Organize the retrieved data into a JSON structure suitable for integration.

6. **Final Article Compilation**

   - **Integration**: Merge the fetched papers into the article, adding inline citations and a comprehensive references section.
   - **Output**: Save the finalized articles in the `data/output/` directory.

## Dependencies

The project relies on the following Python libraries:

- **Standard Libraries**:
  - `os`
  - `json`
  - `logging`
  - `asyncio`
  - `argparse`
  - `fnmatch`
  - `random`
  - `time`
  - `datetime`
  - `xml.etree.ElementTree`

- **Third-Party Libraries**:
  - [`python-dotenv`](https://pypi.org/project/python-dotenv/): For loading environment variables.
  - [`pydantic`](https://pydantic-docs.helpmanual.io/): For data validation and settings management using Python type annotations.
  - [`rich`](https://rich.readthedocs.io/en/stable/): For enhanced logging and terminal output.
  - [`aiohttp`](https://aiohttp.readthedocs.io/en/stable/): For asynchronous HTTP requests.
  - [`google-generativeai`](https://pypi.org/project/google-generativeai/): For interacting with generative AI models.

Install all dependencies using:

```bash
pip install -r requirements.txt
```

*Ensure that your `requirements.txt` includes all necessary packages with appropriate versions.*

## Notes

- **UpToDate Articles**: The pipeline requires access to local UpToDate articles. Users without access should adapt the data preparation step accordingly or skip it, acknowledging that the generated articles may lack certain enhancements.
  
- **API Keys**: Ensure that the `.env` file contains valid API keys for the generative AI model and Semantic Scholar. Missing or invalid keys will prevent the pipeline from functioning correctly.

- **Concurrency Control**: The pipeline uses asynchronous programming and concurrency controls (semaphores) to manage multiple tasks efficiently without overwhelming external APIs.

- **Logging**: Logs are saved in `data/output/pipeline.log` and also displayed in the console with enhanced formatting using the `rich` library. Review logs to monitor the pipeline's progress and troubleshoot issues.

- **Error Handling**: The pipeline includes robust error handling with retries and exponential backoff for network requests, ensuring resilience against transient failures.


## License

This project is licensed under the [MIT License](LICENSE).
