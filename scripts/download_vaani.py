import os

# cache_dir doesn't work wtf. need to set this explicity. 
# Also note that I'm using compute machines from ml.azure.com which has temporary disks. 
# This needs to be set before import huggingface datasets. 
os.environ['HF_HOME'] = '/mnt/hf_cache'
os.environ['HF_DATASETS_CACHE']  = '/mnt/hf_cache'
_ROOT_DIR = os.path.expanduser('~') + "/pshishodia/vaani"
_TRANSCRIPTIONS_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'transcriptions')
_IMAGES_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'images')
_NUM_RETRIES=10

if not os.path.exists(_TRANSCRIPTIONS_OUTPUT_DIR):
    os.makedirs(_TRANSCRIPTIONS_OUTPUT_DIR)
    
if not os.path.exists(_IMAGES_OUTPUT_DIR):
    os.makedirs(_IMAGES_OUTPUT_DIR)

import datasets
from datasets import load_dataset
import os
from datasets import Image
from PIL import Image as PILImage
from io import BytesIO
from tqdm import tqdm
import pandas as pd
import time
import re


if not os.path.exists('/mnt/hf_cache'):
    os.system('sudo mkdir /mnt/hf_cache')

os.system("sudo chmod 777 /mnt/hf_cache/")

_DISTRICTS = ["AndhraPradesh_Anantpur", "AndhraPradesh_Chittoor", "AndhraPradesh_Guntur", "AndhraPradesh_Krishna",
              "AndhraPradesh_Srikakulam", "AndhraPradesh_Vishakapattanam", "Bihar_Araria", "Bihar_Begusarai", "Bihar_Bhagalpur", 
              "Bihar_Darbhanga", "Bihar_EastChamparan", "Bihar_Gaya", "Bihar_Gopalganj", "Bihar_Jahanabad", "Bihar_Jamui", 
              "Bihar_Kishanganj", "Bihar_Lakhisarai", "Bihar_Madhepura", "Bihar_Muzaffarpur", "Bihar_Purnia", "Bihar_Saharsa", 
              "Bihar_Samastipur", "Bihar_Saran", "Bihar_Sitamarhi", "Bihar_Supaul", "Bihar_Vaishali", "Chhattisgarh_Balrampur", 
              "Chhattisgarh_Bastar", "Chhattisgarh_Bilaspur", "Chhattisgarh_Jashpur", "Chhattisgarh_Kabirdham", "Chhattisgarh_Korba", 
              "Chhattisgarh_Raigarh", "Chhattisgarh_Rajnandgaon", "Chhattisgarh_Sarguja", "Chhattisgarh_Sukma", "Goa_NorthSouthGoa",
              "Jharkhand_Jamtara", "Jharkhand_Sahebganj", "Karnataka_Belgaum", "Karnataka_Bellary", "Karnataka_Bijapur",
              "Karnataka_Chamrajnagar", "Karnataka_DakshinKannada", "Karnataka_Dharwad", "Karnataka_Gulbarga", "Karnataka_Mysore",
              "Karnataka_Raichur", "Karnataka_Shimoga", "Maharashtra_Aurangabad", "Maharashtra_Chandrapur", "Maharashtra_Dhule",
              "Maharashtra_Nagpur", "Maharashtra_Pune", "Maharashtra_Sindhudurga", "Maharashtra_Solapur", "Rajasthan_Churu", 
              "Rajasthan_Nagaur", "Telangana_Karimnagar", "Telangana_Nalgonda", "UttarPradesh_Budaun", "UttarPradesh_Deoria", 
              "UttarPradesh_Etah", "UttarPradesh_Ghazipur", "UttarPradesh_Gorakhpur", "UttarPradesh_Hamirpur", "UttarPradesh_Jalaun", 
              "UttarPradesh_JyotibaPhuleNagar", "UttarPradesh_Muzzaffarnagar", "UttarPradesh_Varanasi", "Uttarakhand_TehriGarhwal", 
              "Uttarakhand_Uttarkashi", "WestBengal_DakshinDinajpur", "WestBengal_Jalpaiguri", "WestBengal_Jhargram", "WestBengal_Kolkata", 
              "WestBengal_Malda", "WestBengal_North24Parganas", "WestBengal_PaschimMedinipur", "WestBengal_Purulia",]

# Hide progress bars to avoid clutter.
datasets.disable_progress_bars()

def download_images():
    vaani_images = load_dataset("ARTPARK-IISc/VAANI", "images", num_proc=16, token=os.environ.get("HF_TOKEN"), cache_dir="/mnt/hf_cache")
    vaani_images = vaani_images.cast_column("image", Image(decode=False))
        
    from multiprocessing import Pool

    def save_image(img):
        PILImage.open(BytesIO(img['bytes'])).save(os.path.join(_ROOT_DIR, img['path']))

    with Pool(16) as p:
        list(tqdm(p.imap(save_image, vaani_images['train']['image']), total=len(vaani_images['train']['image'])))

def clean_transcription(text):
    # Remove text within curly braces { ... }
    text = re.sub(r'\{.*?\}', '', text)
    
    # Remove text within angle brackets < ... >
    text = re.sub(r'<.*?>', '', text)
    
    # remove text within closed brackets [ ... ]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Optionally, normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text.endswith('--'):
        text = text[:-2].strip()

    return text

# Some columns aren't present in some subsets. For example: 
# 1. languagesKnown not present in all. Rajasthan_Nagaur
# 2. pincode not present in UttarPradesh_Etah
_RELEVANT_COLUMNS = ['language', 'gender', 'state', 'district', 'isTranscriptionAvailable', 'transcript', 'referenceImage']
for district in tqdm(_DISTRICTS, desc = 'Districts'):
    output_file = os.path.join(_TRANSCRIPTIONS_OUTPUT_DIR, f"{district}.csv")
    if os.path.exists(output_file):
        continue

    ds = None
    for _ in range(_NUM_RETRIES):
        try:
            ds = load_dataset("ARTPARK-IISc/VAANI", district, columns=_RELEVANT_COLUMNS, num_proc=16, token=os.environ.get("HF_TOKEN"), cache_dir="/mnt/hf_cache")
        except Exception as e:
            print(f"Failed!! : {e}\n-------Sleeping for 10s-------")
            time.sleep(10)
    ds = ds.filter(lambda x : x['isTranscriptionAvailable'] == 'Yes')
    transcriptions = [el for el in ds['train']]
    
    pd.DataFrame(transcriptions).to_csv(output_file)
    os.system("rm -rf /mnt/hf_cache/hub/datasets--ARTPARK-IISc--VAANI/*")
    
    
## Save to transcriptions.csv 
dfs = []
for f in os.listdir(_TRANSCRIPTIONS_OUTPUT_DIR):
    df = pd.read_csv(os.path.join(_TRANSCRIPTIONS_OUTPUT_DIR, f))
    df['district'] = f.split('.')[0]
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)
df['clean_transcript'] = df['transcript'].apply(clean_transcription)
df.to_csv(os.path.join(_ROOT_DIR, 'transcriptions.csv'), index=False)