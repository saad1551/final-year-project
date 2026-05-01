"""Quick sanity check that the google-genai SDK + url_context tool work.
Read the API key from env (set JUDGE_API_KEY=... before running)."""
import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig

api_key = os.environ.get("JUDGE_API_KEY")
if not api_key:
    raise RuntimeError(
        "JUDGE_API_KEY env var not set. "
        "Get a key at https://aistudio.google.com/apikey."
    )
client = genai.Client(api_key=api_key)
model_id = "gemini-2.5-flash"

tools = [
  {"url_context": {}},
]

url1 = "https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592"
url2 = "https://www.allrecipes.com/recipe/21151/simple-whole-roast-chicken/"

response = client.models.generate_content(
    model=model_id,
    contents=f"Compare the ingredients and cooking times from the recipes at {url1} and {url2}",
    config=GenerateContentConfig(
        tools=tools,
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)

# For verification, you can inspect the metadata to see which URLs the model retrieved
print(response.candidates[0].url_context_metadata)