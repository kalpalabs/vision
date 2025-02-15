
from PIL import Image
import requests
from transformers import SiglipProcessor, SiglipModel, SiglipImageProcessor, SiglipTokenizer, AutoModel
import torch
from min_siglip import SiglipVisionModel, SiglipVisionConfig, SiglipTextConfig, SiglipTextModel

torch.set_grad_enabled(False)
_DEMO_IMAGE = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
_DEMO_TEXTS = ["a photo of 2 cats", "a photo of 2 dogs"]

def get_new_key(k):
    k = k.replace('encoder.', '')
    k = k.replace('embeddings.', '')
    while k.startswith('.'):
        k = k[1:]
    return k

def get_new_vision_model(hf_model):
    # Remove keys that aren't present in our SiglipVisionConfig
    valid_vision_config = {k: v for k, v in hf_model.config.vision_config.to_dict().items() if k in SiglipVisionConfig.__dataclass_fields__.keys()}
    vision_config = SiglipVisionConfig(**valid_vision_config)
    
    # Initialize the model. 
    new_vision_model = SiglipVisionModel(config=vision_config)

    # Load weights. 
    new_vision_model.load_state_dict({get_new_key(k): v for k, v in hf_model.vision_model.state_dict().items()})
    return new_vision_model

def get_new_text_model(hf_model):
    # Remove keys that aren't present in our SiglipVisionConfig
    valid_text_config = {k: v for k, v in hf_model.config.text_config.to_dict().items() if k in SiglipTextConfig.__dataclass_fields__.keys()}
    text_config = SiglipTextConfig(**valid_text_config)
    
    # Initialize the model. 
    new_text_model = SiglipTextModel(config=text_config)

    # Load weights. 
    new_text_model.load_state_dict({get_new_key(k): v for k, v in hf_model.text_model.state_dict().items()})
    return new_text_model

# Verify that the new siglip implementation generates the same output as the huggingface's siglip implementation. 
def test_new_vision_implementation_correctness(model_name:str=""):
    print(f"========= Verifying New Vision Implementation: {model_name} ==========")
    hf_model = SiglipModel.from_pretrained(model_name)
    hf_image_processor = SiglipImageProcessor(model_name)
    new_vision_mdoel = get_new_vision_model(hf_model)
    
    pixel_values = hf_image_processor(_DEMO_IMAGE, return_tensors="pt")['pixel_values']
    
    hf_output = hf_model.vision_model(pixel_values)
    new_output = new_vision_mdoel(pixel_values)
    
    assert torch.equal(new_output['last_hidden_state'], hf_output.last_hidden_state)
    assert torch.equal(new_output['pooler_output'], hf_output.pooler_output)
    
    print(f"========= New Vision Implementation VERIFIED: {model_name} ==========")
    

# Verify that the new siglip implementation generates the same output as the huggingface's siglip implementation. 
def test_new_text_implementation_correctness(model_name:str=""):
    print(f"========= Verifying New Text Implementation : {model_name} ==========")
    hf_model = SiglipModel.from_pretrained(model_name)
    hf_tokenizer = SiglipTokenizer.from_pretrained(model_name)
    new_text_model = get_new_text_model(hf_model)
    
    input_ids = hf_tokenizer(_DEMO_TEXTS, return_tensors="pt", padding="max_length")['input_ids']

    hf_output = hf_model.text_model(input_ids)
    new_output = new_text_model(input_ids)
    
    assert torch.equal(new_output['last_hidden_state'], hf_output.last_hidden_state)
    assert torch.equal(new_output['pooler_output'], hf_output.pooler_output)
    
    print(f"========= New Text Implementation VERIFIED: {model_name} ==========")

# Verify for different model sizes, to ensure that some constants haven't been hardcoded. 
test_new_vision_implementation_correctness(model_name="google/siglip-base-patch16-224")
test_new_text_implementation_correctness(model_name="google/siglip-base-patch16-224")