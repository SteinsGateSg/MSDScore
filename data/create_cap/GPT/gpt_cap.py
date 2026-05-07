import os
import json
import time
import base64
import mimetypes
import glob
from tqdm import tqdm

import openai
from openai import OpenAI

IMAGE_DIR = os.environ.get("IMAGE_DIR", "data/images/val2017")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/coco_descriptions_gpt4omini")
MAX_IMAGES = 5000
SAVE_INTERVAL = 100

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_TEXT = "Describe the main subject in one concise English sentence."
MAX_TOKENS = 80

SKIP_ERROR = False
# ======================================


def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_to_data_url(path: str) -> str:
    mime = guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_with_retry(fn, *, max_retries=6, base_sleep=1.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt))
        except openai.APIStatusError as e:
            if e.status_code and e.status_code >= 500:
                last_err = e
                time.sleep(base_sleep * (2 ** attempt))
            else:
                raise
    raise last_err


def caption_image(client: OpenAI, image_path: str) -> str:
    data_url = image_to_data_url(image_path)

    def _do():
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    return call_with_retry(_do)


def load_latest_results(output_dir: str) -> tuple[dict, str | None]:
    results = {}
    step_files = glob.glob(os.path.join(output_dir, "results_step_*.json"))
    if not step_files:
        return results, None

    def step_num(p: str) -> int:
        # results_step_1900.json -> 1900
        base = os.path.basename(p)
        n = base.split("_")[-1].split(".")[0]
        return int(n)

    latest = max(step_files, key=step_num)
    with open(latest, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, latest


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("export OPENAI_API_KEY=YOUR_API_KEY")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    all_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    if MAX_IMAGES:
        all_files = all_files[:MAX_IMAGES]
    print(f"Found {len(all_files)} images to process.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results, latest_path = load_latest_results(OUTPUT_DIR)
    if latest_path:
        print(f"Resuming from {latest_path}, loaded {len(results)} captions.")
    else:
        print("No existing results found, starting fresh.")

    if SKIP_ERROR:
        done = set(results.keys())
    else:
        done = {k for k, v in results.items() if isinstance(v, str) and not v.startswith("ERROR:")}

    todo_files = [f for f in all_files if f not in done]
    print(f"Remaining: {len(todo_files)} images (done={len(done)}).")

    start_time = time.time()
    pbar = tqdm(total=len(todo_files), desc="Generating (resume)")

    for processed_count, filename in enumerate(todo_files, start=1):
        img_path = os.path.join(IMAGE_DIR, filename)
        try:
            results[filename] = caption_image(client, img_path)
        except Exception as e:
            results[filename] = f"ERROR: {e}"
            print(f"\nError processing {filename}: {e}")

        pbar.update(1)

        if (len(results) % SAVE_INTERVAL == 0) or (len(results) == len(all_files)):
            save_path = os.path.join(OUTPUT_DIR, f"results_step_{len(results)}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix({"Speed": f"{speed:.2f} img/s", "Saved": os.path.basename(save_path)})

    pbar.close()

    final_path = os.path.join(OUTPUT_DIR, f"results_step_{len(results)}.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"All done. Final results saved to {final_path}")


if __name__ == "__main__":
    main()
