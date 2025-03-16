import os
os.environ['hf_HUB_ENABLE_HF_TRANSFER'] = '1' # faster downloading for hf models/datasets.

from vllm import LLM, SamplingParams
import io
import base64
from PIL import Image

def encode_image(img):
    mime_type, out_format = "image/jpeg", "jpeg"
    buffered = io.BytesIO()
    img.save(buffered, format=out_format)
    img_byte_data = buffered.getvalue()

    return base64.b64encode(img_byte_data).decode('utf-8')

if __name__ == "__main__":
    llm = LLM(model="google/gemma-3-27b-it", max_model_len=1024, max_seq_len_to_capture=1024, enable_prefix_caching=False)
        
    _SYSTEM_MESSAGE = {
        "role": "system",
        "content": [
            {"type": "text", "text": '''Caption the ImageNet-21K in a single, detailed paragraph under 20 words, without introductory phrases like "This image showcases", "Here's a detailed description".'''},
        ],
    }

    message = [
        _SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(Image.open('/home/azureuser/pshishodia/waterweed.png'))}"
                    }
                },
                {"type": "text", "text": "ImageNet-21K Labels: waterweed."},
            ],
        },
    ]

    sampling_params = SamplingParams(max_tokens=64, min_p=0.05)

    outputs = llm.chat([message] * 24, sampling_params=sampling_params)
    print("Outputs: ", [output.outputs[0].text for output in outputs])