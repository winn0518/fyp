from ibm_watsonx_ai.foundation_models import Model
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("PROJECT_ID")
api_key = os.getenv("WATSONX_APIKEY")
url = os.getenv("WATSONX_URL")

if not api_key or not project_id:
    raise ValueError("❌ WatsonX credentials missing in .env")

def ask_watsonx(prompt: str):
    model = Model(
        model_id="mistralai/mistral-medium-2505",
        params={"decoding_method": "greedy"},
        credentials={"apikey": api_key, "url": url},
        project_id=project_id,
    )

    response = model.generate(prompt=prompt)
    return response["results"][0]["generated_text"]
