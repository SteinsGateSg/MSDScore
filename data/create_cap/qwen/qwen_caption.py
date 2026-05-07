import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import time

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen-VL-Chat")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images/val2017")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/coco_descriptions_qwen")
MAX_IMAGES = 5000
SAVE_INTERVAL = 5000

print(f"> Loading Qwen-VL model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda", trust_remote_code=True).eval()

model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

model.generation_config.max_new_tokens = 80
model.generation_config.do_sample = False  
model.generation_config.top_p = 0.01       

print("Model loaded.")


all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
if MAX_IMAGES:
    all_files = all_files[:MAX_IMAGES]
print(f"Found {len(all_files)} images to process.")


os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}
PROMPT_TEXT = "Describe the main subject in one concise English sentence."

start_time = time.time()
pbar = tqdm(total=len(all_files), desc="Generating")


for i, filename in enumerate(all_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    
    try:
        query = tokenizer.from_list_format([
            {'image': img_path},
            {'text': PROMPT_TEXT},
        ])
        
        response, _ = model.chat(tokenizer, query=query, history=None)
        
        results[filename] = response.strip()

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        results[filename] = f"ERROR: {e}"
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()

    pbar.update(1)

    if (i + 1) % SAVE_INTERVAL == 0 or i == len(all_files) - 1:
        save_path = os.path.join(OUTPUT_DIR, f"results_step_{i+1}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        pbar.set_postfix({"Speed": f"{speed:.2f} img/s"})

pbar.close()
print(f"All done. Results saved to {OUTPUT_DIR}")
