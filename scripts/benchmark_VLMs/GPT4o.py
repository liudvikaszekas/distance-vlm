from openai import OpenAI
import json
import os
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import base64
import json 

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  
def predict_relation(image_path, prompt):
    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )

    return completion.choices[0].message.content

# TODO: exchange to the dataset
with open("dataset.json", 'r') as f:
    data = json.load(f)

results = {}
for image_path in os.listdir("images"):
    if image_path.endswith((".jpg", ".png", ".jpeg")):
        prompt = data[image_path]['question']
        result = predict_relation(f"images/{image_path}", prompt)
        results[image_path] = (prompt, result)

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
        