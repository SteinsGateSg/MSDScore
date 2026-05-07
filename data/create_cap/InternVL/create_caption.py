import json
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import time

MODEL_PATH = os.environ.get("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL2-8B")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images/val2017")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/coco_descriptions_internvl")
MAX_IMAGES = 1000
BATCH_SIZE = 8   
SAVE_INTERVAL = 200 
# =========================================

# --------------------------------------------------------
# --------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
# --------------------------------------------------------

print("> Loading InternVL2-8B model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).eval().cuda()
print("Model loaded.")

all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
all_files = all_files[:MAX_IMAGES]
print(f"Found {len(all_files)} images.")

generation_config = dict(max_new_tokens=80, do_sample=False)
PROMPT = "<image>\nDescribe the main subject in one concise English sentence."

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}
batch_buffer = [] 
pixel_values_buffer = [] 
num_patches_buffer = [] 

start_time = time.time()
pbar = tqdm(total=len(all_files), desc="Batch Processing")

for i, filename in enumerate(all_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    
    try:
        pv = load_image(img_path, max_num=6).to(torch.bfloat16).cuda()
        
        pixel_values_buffer.append(pv)
        num_patches_buffer.append(pv.size(0)) 
        batch_buffer.append(filename)
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        results[filename] = "ERROR_LOAD"

    if len(batch_buffer) == BATCH_SIZE or i == len(all_files) - 1:
        if not batch_buffer: continue 
        
        try:
            pixel_values_batch = torch.cat(pixel_values_buffer, dim=0)
            
            questions = [PROMPT] * len(batch_buffer)
            
            responses = model.batch_chat(
                tokenizer, 
                pixel_values_batch, 
                num_patches_list=num_patches_buffer, 
                questions=questions, 
                generation_config=generation_config
            )
            
            for fname, resp in zip(batch_buffer, responses):
                clean_resp = resp.replace("<image>\n", "").strip()
                if "concise English sentence" in clean_resp: 
                    clean_resp = clean_resp.split("concise English sentence")[-1].strip()
                results[fname] = clean_resp
                
        except Exception as e:
            print(f"Batch inference failed: {e}")
            for fname in batch_buffer:
                results[fname] = f"ERROR_INFERENCE: {e}"
        
        batch_buffer = []
        pixel_values_buffer = []
        num_patches_buffer = []
        
        pbar.update(BATCH_SIZE if i < len(all_files)-1 else (i % BATCH_SIZE) + 1)
        
        if (i + 1) % SAVE_INTERVAL == 0 or i == len(all_files) - 1:
            save_path = os.path.join(OUTPUT_DIR, f"results_step_{i+1}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            elapsed = time.time() - start_time
            print(f"\nSaved {save_path}. Speed: {len(results)/elapsed:.2f} imgs/sec")

pbar.close()
print("All done.")
