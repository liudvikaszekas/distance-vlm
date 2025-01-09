from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json
import os
from tqdm import tqdm
from PIL import Image

# TODO: Change this for dlab setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
low_cpu_mem_usage = True

# Load the BLIP2 processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Paths and filenames
model_name = "BLIP2"
meta_data = "mm-vet-v2.json"
result_path = "results/"
image_folder = "images_center/"
results_path = os.path.join(result_path, f"{model_name}.json")

# Load the dataset (questions, etc.)
with open(meta_data, 'r') as f:
    data = json.load(f)

# Load results if they already exist, else initialize an empty dict
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = {}

# Batch size for batching inference (set based on your memory constraints)
batch_size = 8

# List to accumulate inputs for batching
batch_images = []
batch_questions = []
batch_ids = []

# Process each item in the dataset
for i in tqdm(range(len(data))):
    id = f"v2_{i}"

    # Skip if the result is already calculated
    if id in results:
        continue

    # Extract question and associated image paths
    question = data[id]["question"].strip()
    queries = question.split("<IMG>")
    image = None

    # Find and open the image corresponding to the query
    for query in queries:
        query = query.strip()
        if query.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, query)
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            break  # Only process the first image found in queries
    
    if image is None:
        print(f"No image found for question ID {id}")
        continue

    # Accumulate the question and image for batching
    batch_images.append(image)
    batch_questions.append(question)
    batch_ids.append(id)

    # Once the batch is full, process it
    if len(batch_images) == batch_size:
        # Prepare inputs and move to the device
        inputs = processor(images=batch_images, text=batch_questions, return_tensors="pt", padding=True).to(device)

        # Use mixed precision for inference
        with torch.cuda.amp.autocast(enabled=device == "cuda"):
            outputs = model.generate(**inputs)

        # Decode and store the results
        decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        for j, decoded_output in enumerate(decoded_outputs):
            results[batch_ids[j]] = decoded_output

        # Clear the batch
        batch_images = []
        batch_questions = []
        batch_ids = []

        # Periodically save results
        if i % 100 == 0:  # Save after every 100 examples
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

# Process any remaining examples in the final batch
if batch_images:
    inputs = processor(images=batch_images, text=batch_questions, return_tensors="pt", padding=True).to(device)
    with torch.autocast("cuda"):
        outputs = model.generate(**inputs)
    decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
    for j, decoded_output in enumerate(decoded_outputs):
        results[batch_ids[j]] = decoded_output

# Final save
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("Processing complete.")
