import os
import base64
import json
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
credentials_base64 = os.getenv("SERVICE_ACCOUNT_CREDENTIALS_BASE64")

decoded_credentials = base64.b64decode(credentials_base64).decode('utf-8')
credentials_info = json.loads(decoded_credentials)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Initialize VertexAI and AIP
aiplatform.init(credentials=credentials, project='rbio-p-datasharing')
vertexai.init(project="rbio-p-datasharing", location="us-west1")
