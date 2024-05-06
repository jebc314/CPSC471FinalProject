from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import urllib.request
import json
import os
import torch
import sys

# Load iamges

input_dir = "input"
images_dir = input_dir + "\\images\\"

with urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json") as url:
    imagenet_classes_dict = json.load(url)
with urllib.request.urlopen("https://raw.githubusercontent.com/LUSSeg/ImageNet-S/main/data/categories/ImageNetS_categories_im50.txt") as url:
    imagenetS50_ids_dict = {str(x).replace("b'", "").replace("\\n'", "").replace("'",""):i+1 for i, x in enumerate(url)}

def images_list(image_paths):
    images = []
    for image_path in image_paths:
        for image_name in os.listdir(image_path):
            image = Image.open(image_path + image_name)
            if image.mode == 'L':
                image = image.convert(mode='RGB')
            images.append(image)
    return images

all_images_original = images_list([images_dir + classid + "\\" for classid in imagenetS50_ids_dict])[10*int(sys.argv[1]):10*(int(sys.argv[1]) + 1)]

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.eval();

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 100 uses ~ 16 gb
inputs = processor(images=all_images_original, return_tensors="pt").to(device)
outputs = model.to(device)(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
_, preds = torch.max(logits, 1)

torch.save(inputs['pixel_values'], f"ImageNet_VT/proper_data{sys.argv[1]}.pt")
torch.save(preds, f"ImageNet_VT/preds-{sys.argv[1]}.pt")