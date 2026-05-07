import json
import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoImageProcessor, LlavaProcessor, AutoTokenizer
from tqdm import tqdm
import time

MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH", "llava-hf/llava-1.5-7b-hf")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images/val2017")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/coco_descriptions_llava")
MAX_IMAGES = 5000  
SAVE_INTERVAL = 5000 
# =========================================

# --------------------------------------------------------
# --------------------------------------------------------
print(f"> Loading LLaVA-1.5 model from {MODEL_PATH}...")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto")
model.eval()

print(f"Model loaded on {model.device}.")

# --------------------------------------------------------
# --------------------------------------------------------
all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
if MAX_IMAGES:
    all_files = all_files[:MAX_IMAGES]
print(f"Found {len(all_files)} images to process.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}

PROMPT_TEXT = "USER: <image>\nDescribe the main subject in the image in one concise English sentence.\nASSISTANT:"

# --------------------------------------------------------
# --------------------------------------------------------
start_time = time.time()
pbar = tqdm(total=len(all_files), desc="Generating")

for i, filename in enumerate(all_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    
    try:
        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(text=PROMPT_TEXT, images=raw_image, return_tensors="pt")
        
        inputs = {k: v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=100, 
                do_sample=False,    
                # temperature=0.7,
            )

        decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        full_response = decoded_text.strip()
        if "ASSISTANT:" in full_response:
            clean_response = full_response.split("ASSISTANT:")[-1].strip()
        else:
            clean_response = full_response
            
        results[filename] = clean_response

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        results[filename] = f"ERROR: {e}"
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()

    pbar.update(1)
    elapsed = time.time() - start_time
    speed = (i + 1) / elapsed
    pbar.set_postfix({"Speed": f"{speed:.2f} img/s"})

    if (i + 1) % SAVE_INTERVAL == 0 or i == len(all_files) - 1:
        save_path = os.path.join(OUTPUT_DIR, f"results_step_{i+1}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

pbar.close()
print(f"All done. Results saved to {OUTPUT_DIR}")
