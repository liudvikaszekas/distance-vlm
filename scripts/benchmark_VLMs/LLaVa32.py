from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForPreTraining, MllamaForConditionalGeneration
from PIL import Image
import os
import json
import requests
import torch
from tqdm import tqdm


# TODO: Change this for dlab setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
low_cpu_mem_usage = True

# TODO: Create a token for the Hugging Face and add to .env file
login(token = os.environ["HF_TOKEN"])

model_name = "llava32_448"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
image_folder = "img_mmvet/images_448/"
meta_data = "mm-vet-v2.json"
result_path = "results/"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.tie_weights()
processor = AutoProcessor.from_pretrained(model_id)

results_path = os.path.join(result_path, f"{model_name}.json")
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = {}

with open(meta_data, 'r') as f:
    data = json.load(f)

for i in tqdm(range(len(data))):
    id = f"v2_{i}"
    
    if id in results:
        continue

    question = data[id]["question"].strip()
    queries = question.split("<IMG>")
    img_num = 0

    for query in queries:
        query = query.strip()
        if query == "":
            continue
        if query.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, query)
            image = Image.open(image_path)

    messages = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": queries[0]},
            {"type": "image"},
            ],
        },
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=28000)
    results[id] = processor.decode(output[0])

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
  

