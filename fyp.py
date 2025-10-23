from dotenv import load_dotenv  # import the package
import os

# Load variables from your .env file
load_dotenv()

# Access your Replicate API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
print(REPLICATE_API_TOKEN)  # optional: check it works
