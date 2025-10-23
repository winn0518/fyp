from langchain_replicate import ChatReplicate
from dotenv import load_dotenv
import os

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

model_path = "ibm-granite/granite-4.0-h-small"
model = ChatReplicate(
    model=model_path,
    replicate_api_token=REPLICATE_API_TOKEN,
    model_kwargs={"max_tokens": 1000, "min_tokens": 100},
)

from langchain_core.prompts import ChatPromptTemplate

query = "Who won in the Pantoja vs Asakura fight at UFC 310?"
prompt_template = ChatPromptTemplate.from_template("{input}")
chain = prompt_template | model
output = chain.invoke({"input": query})
print(output.text())

