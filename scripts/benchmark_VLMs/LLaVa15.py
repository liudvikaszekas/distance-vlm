from transformers import pipeline, BitsAndBytesConfig, AutoProcessor
import torch
import requests
from PIL import Image
import os
import json 
import base64
from tqdm import tqdm


# TODO: Change this for dlab setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
low_cpu_mem_usage = True

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "llava1.5"
model_id = "llava-hf/llava-1.5-7b-hf"
image_folder = "images/"
meta_data = "mm-vet-v2.json"
result_path = "results/"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
results_path = os.path.join(result_path, f"{model_name}.json")
processor = AutoProcessor.from_pretrained(model_id)

with open(meta_data, 'r') as f:
    data = json.load(f)

if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = {}

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

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": queries[0]},
            {"type": "image"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    response = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    results[id] = response

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)