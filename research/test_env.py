from dotenv import load_dotenv
import os

# Load .env file (make sure it's in the same folder)
load_dotenv()

# Get your API key
api_key = os.getenv("GOOGLE_API_KEY")  # replace with the actual name in .env
print("API key loaded:", api_key)
