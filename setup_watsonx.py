from ibm_watsonx_ai.foundation_models import Model
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("WATSONX_PROJECT_ID")
api_key = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")

if not api_key or not project_id:
    raise ValueError("❌ WatsonX credentials missing in .env")

def ask_watsonx(prompt: str):
    model = Model(
        model_id="ibm/granite-13b-chat-v2",
        params={"decoding_method": "greedy", "max_new_tokens": 200},
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=project_id,
    )

    response = model.generate_text(prompt=prompt)
    return response
