from openai import OpenAI
import base64
import os
from datasets import load_dataset
import io
import json
from tqdm import tqdm
import asyncio


## Constants.
_SYSTEM_PROMPT='''Caption the ImageNet-21K in a single, detailed paragraph, without introductory phrases like "This image showcases", "Here's a detailed description".'''
_MODEL_ID = 'gemini-2.0-flash'
_ROOT_CAPTIONS_DIR = 'imagenet-1k-captions'

assert os.environ.get('GEMINI_API_KEY'), "Gemini API Key is not present!"
if not os.path.exists(_ROOT_CAPTIONS_DIR):
    os.makedirs(_ROOT_CAPTIONS_DIR)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")  # Or "JPEG", "GIF", etc., depending on your needs
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_messages(img, wordnet_lemmas, system_prompt):
  messages=[
    {
        "role" : "system",
        "content" : [
            {"type": "text", "text": system_prompt},
        ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{encode_image_from_pil(img)}"
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
    with open('imagenet-1k-synsets.txt') as f:
        for line in f.readlines():
            wordnet_id, lemma = line.split(":")[0].strip(), line.split(":")[1].strip()
            wordnet_ids_to_lemmas[wordnet_id] = lemma
    return wordnet_ids_to_lemmas

def get_shard_str(n):
  n = str(n)
  return '0' * (4 - len(n)) + n

async def get_api_response(inputs):
    wordnet_lemmas = wordnet_ids_to_lemmas[inputs['__key__'].split('_')[0]]

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
    except Exception as e:
        print("FAILED: ", e)
    return ''

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
wordnet_ids_to_lemmas = get_wordnet_id_to_lemmas()

async def process_shard(shard: int):
    output_filename = os.path.join(_ROOT_CAPTIONS_DIR, f"captions-{get_shard_str(shard)}.json")
    # pass if we fetched captions for this shard, before. 
    if os.path.exists(output_filename):
        return

    ds_shard = load_dataset(
        "timm/imagenet-1k-wds",
        data_files=f"imagenet1k-train-{get_shard_str(shard)}.tar",
        split="train"
    )
    
    tasks = []
    for inputs in ds_shard:
        tasks.append(get_api_response(inputs))

    # Use asyncio.gather to run all requests concurrently.
    responses = await asyncio.gather(*tasks)

    key_to_captions = {}
    for inputs, response in zip(ds_shard, responses):
        key_to_captions[inputs['__key__']] = response

    with open(output_filename, 'w') as f:
        json.dump(key_to_captions, f, indent=4)
        
    # wait if previously a lot of queries failed. 
    percent_queries_failed = len([1 for resp in responses if not resp]) / len(responses)
    sleep_time = percent_queries_failed * 60
    print(f"Sleeping for {sleep_time=}")
    await asyncio.sleep(sleep_time)
        
async def main():
    NUM_SHARDS = 1024
    for shard in tqdm(range(NUM_SHARDS), desc="ShardsCompleted"):
        await process_shard(shard)

if __name__ == "__main__":
    asyncio.run(main())
