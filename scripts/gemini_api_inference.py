import os
import base64
import json
import io
import random
import time
from tqdm import tqdm
from datasets import load_dataset
import concurrent.futures
from openai import OpenAI
from openai import RateLimitError

## Constants.
_SYSTEM_PROMPT = '''Caption the ImageNet-21K in a single, detailed paragraph, without introductory phrases like "This image showcases", "Here's a detailed description".'''
_MODEL_ID = 'gemini-2.0-flash'
_ROOT_CAPTIONS_DIR = 'imagenet-1k-captions'

assert os.environ.get('GEMINI_API_KEY'), "Gemini API Key is not present!"
if not os.path.exists(_ROOT_CAPTIONS_DIR):
    os.makedirs(_ROOT_CAPTIONS_DIR)

def encode_image_from_pil(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")  # Adjust format if needed.
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_messages(img, wordnet_lemmas, system_prompt):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_from_pil(img)}"
                    },
                },
                {
                    "type": "text",
                    "text": f"ImageNet-21K Wordnet Labels: {wordnet_lemmas}.",
                },
            ],
        }
    ]
    return messages

def get_wordnet_id_to_lemmas():
    wordnet_ids_to_lemmas = {}
    # Synsets are copied from https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php
    with open('imagenet-1k-synsets.txt') as f:
        for line in f.readlines():
            parts = line.split(":")
            if len(parts) < 2:
                continue
            wordnet_id, lemma = parts[0].strip(), parts[1].strip()
            wordnet_ids_to_lemmas[wordnet_id] = lemma
    return wordnet_ids_to_lemmas

def get_shard_str(n):
    n = str(n)
    return '0' * (4 - len(n)) + n

def get_api_response(inputs):
    wordnet_lemmas = wordnet_ids_to_lemmas[inputs['__key__'].split('_')[0]]
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=_MODEL_ID,
                messages=get_messages(img=inputs['jpg'], wordnet_lemmas=wordnet_lemmas, system_prompt=_SYSTEM_PROMPT)
            )
            resp = response.choices[0].message
            content = resp.content.strip()
            assert resp.refusal is None, f"Refusal is not None: {resp}"
            assert content, f"Content is empty: {content}"
            return content
        except RateLimitError as e:
            sleep_time = int(random.random() * 5 * 100) / 100
            print(f"RATE LIMITED. Sleeping for {sleep_time=}")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"FAILED: {inputs['__key__']=} \nError:", e)
            return ''

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
wordnet_ids_to_lemmas = get_wordnet_id_to_lemmas()

def process_shard(shard: int):
    output_filename = os.path.join(_ROOT_CAPTIONS_DIR, f"captions-{get_shard_str(shard)}.json")
    # Skip processing if output file already exists.
    if os.path.exists(output_filename):
        return

    ds_shard = load_dataset(
        "timm/imagenet-1k-wds",
        data_files=f"imagenet1k-train-{get_shard_str(shard)}.tar",
        split="train"
    )
    
    responses = {}
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_to_key = {
            executor.submit(get_api_response, inputs): inputs['__key__']
            for inputs in ds_shard
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                response = future.result()
            except Exception as e:
                print(f"{key} generated an exception: {e}")
                response = ''
            responses[key] = response

    # Save the responses.
    with open(output_filename, 'w') as f:
        json.dump(responses, f, indent=4)
        
    end_time = time.perf_counter()
    
    MAX_REQUESTS_PER_MIN = 2000
    time_needed = len(ds_shard) / MAX_REQUESTS_PER_MIN * 60
    sleep_time = max(0, time_needed - (end_time - start_time))
    print(f"Sleeping for {sleep_time=}")
    time.sleep(sleep_time)

def main():
    NUM_SHARDS = 1024
    for shard in tqdm(range(NUM_SHARDS), desc="ShardsCompleted"):
        process_shard(shard)

if __name__ == "__main__":
    main()
