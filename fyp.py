from dotenv import load_dotenv
import os

def check_env_keys(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
    else:
        print("✅ All environment variables loaded successfully.")

if __name__ == "__main__":
    load_dotenv()
    check_env_keys(["OPENAI_API_KEY", "WATSONX_URL", "WATSONX_APIKEY", "PROJECT_ID"])
