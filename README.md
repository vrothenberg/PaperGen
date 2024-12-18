# Install

```
mamba create -n papergen python=3.12

pip install python-dotenv

pip install google-cloud-aiplatform

pip install google-auth

pip install pydantic

pip install google-generativeai

```

# Requirements

Create .env file with the following variables:

- GOOGLE_API_KEY
- GOOGLE_PROJECT_ID
- SEMANTIC_SCHOLAR_API_KEY

- Google Cloud Account
- Google Cloud Project
- Google Cloud Project ID
- Google Cloud Region
- Google Cloud Service Account
- Google Cloud Service Account Key

# Setup

UpToDate articles stored in local directory 

Pre-computed BM25 index of most relevant articles for each condition

# Usage 

`python main.py`