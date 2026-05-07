import os
import json
import base64
import mimetypes
import argparse
import requests

def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

def image_to_data_url(path: str) -> str:
    mime = guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def main():
    parser = argparse.ArgumentParser(description="Check gpt-4o-mini API via /v1/chat/completions")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--prompt", default="hello", help="User prompt.")
    parser.add_argument("--image", default=None, help="Optional image path to test vision")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Please `export OPENAI_API_KEY=...`")

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if args.image:
        data_url = image_to_data_url(args.image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
    else:
        messages = [{
            "role": "user",
            "content": args.prompt,
        }]

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0,
    }

    print(f"POST {url}")
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except Exception as e:
        print("Request failed:", repr(e))
        return

    print("HTTP:", r.status_code)
    if r.status_code != 200:
        print("Error response text:")
        print(r.text)
        print("\nTips:")
        return

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = None

    print("Call succeeded.")
    if content is not None:
        print("\n--- model output ---")
        print(content)

    usage = data.get("usage")
    if usage:
        print("\n--- usage ---")
        print(usage)

if __name__ == "__main__":
    main()
