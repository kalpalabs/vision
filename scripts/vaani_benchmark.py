import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from torch.utils.data import Dataset, DataLoader

# only builds with transformer loaded from HEAD.
# !uv pip install git+https://github.com/huggingface/transformers


_ROOT_DIR = os.path.expanduser('~') + "/pshishodia/vaani"
_TRANSCRIPTIONS_FILE = os.path.join(_ROOT_DIR, 'transcripts.csv')
_IMAGES_DIR = os.path.join(_ROOT_DIR, 'images')

def read_images():
    image_filenames = os.listdir(_IMAGES_DIR)
    images = [Image.open(os.path.join(_IMAGES_DIR, f)) for f in tqdm(image_filenames, desc="Reading Images")]
    images = [img if img.mode == 'RGB' else img.convert('RGB') for img in images]
    
    return image_filenames, images

def read_captions():
    transcripts_df = pd.read_csv(_TRANSCRIPTIONS_FILE)
    
    # Surprisingly some transcripts are duplicated for different images. Some easy ones like "यह एक हॉस्पिटल है।" make sense.
    # but some are even more complex. For example "इस इमेज में मुझे एक बहोत बड़ा गार्डन दिख रहा है जिसका मिट्टी है जो" is exactly same 
    # for 4 queries with two different districts, but the same image. Anyway, we remove them all to reduce confusion.
    transcripts_df = transcripts_df.drop_duplicates(subset=['clean_transcript'], keep=False)
    records = transcripts_df[['referenceImage', 'clean_transcript']].to_dict('records')
    
    del transcripts_df
    return [r['clean_transcript'] for r in records], records # captions, records

def get_image_caption_mappings(image_filenames, records):
    filename_to_img_idx = {f:i for i, f in enumerate(image_filenames)}
    caption_to_img_idx = []
    image_to_caption_idxs = [[] for _ in range(len(image_filenames))]
    
    for i, record in enumerate(records):
        img_idx = filename_to_img_idx[record['referenceImage'][len('Images/'):]]
        caption_to_img_idx.append(img_idx)
        image_to_caption_idxs[img_idx].append(i)

    del filename_to_img_idx
    return caption_to_img_idx, image_to_caption_idxs

def sanity_check_image_caption_mapping():
    ## Ideally, we should just write captions on images, but that currently doesn't work and requires Indic fonts.
    caption_idx = 1273
    print("================Caption to Image Mapping================")
    print("Caption: ", captions[caption_idx])
    print("Corresponding Image: img1.jpg")

    img_filename = image_filenames[caption_to_img_idx[caption_idx]]
    Image.open(os.path.join(_IMAGES_DIR, img_filename)).save('img1.jpg')

    img_idx = 1273
    print("================Image to Caption Mapping================")
    print("Image: img2.jpg")
    Image.open(os.path.join(_IMAGES_DIR, image_filenames[img_idx])).save('img2.jpg')
    print("Corresponding Captions: ", "\n".join(str(i) + ". " + captions[idx] for i, idx in enumerate(image_to_caption_idxs[img_idx])))


image_filenames = os.listdir(_IMAGES_DIR)
captions, records = read_captions()
caption_to_img_idx, image_to_caption_idxs = get_image_caption_mappings(image_filenames, records)

sanity_check_image_caption_mapping()

print("========LOADING MODEL============")
ckpt = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)
print("========LOADED MODEL============")


class ImageDataset(Dataset):
    def __init__(self, image_filenames):
        self.image_filenames = image_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(_IMAGES_DIR, self.image_filenames[idx]))
        
        # Some images have CMYK format which has 4 channels instead of 3 for RGB.
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

# Create the dataset
dataset = ImageDataset(image_filenames)
BATCH_SIZE = 32

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=lambda batch: processor(images=batch, return_tensors="pt"),
    shuffle=False,
    num_workers=4,
    pin_memory=True,         # Speeds up transfer to GPU
    prefetch_factor=2        # num batches pre-fetched by each worker
)

# Processing each image takes 8ms so processing all would take (120K * 0.008) / 60 = 17mins.
# using a dataloader with prefetching future batches (prefetch_factor) hides this time by fetching
# future batches in parallel when GPU is processing the current batch. 
image_embeddings_list = []
for batch_inputs in tqdm(dataloader, desc="Create Image Embeddings"):
    batch_inputs = {k: v.to(model.device, non_blocking=True) for k, v in batch_inputs.items()}
    
    with torch.no_grad():
        batch_image_embeddings = model.get_image_features(**batch_inputs)

    image_embeddings_list.append(batch_image_embeddings.cpu())
image_embeddings = torch
