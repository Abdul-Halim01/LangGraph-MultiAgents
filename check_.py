from dotenv import load_dotenv
import os

load_dotenv(override=True)

print("GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))
