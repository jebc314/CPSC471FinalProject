import torch
import os
from PIL import Image
import torchvision.transforms.v2 as transforms
import urllib.request
import json
import numpy as np
from transformers import ViTImageProcessor

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

seg_dir = input_dir + "\\ImageNetS50\\train-semi-segmentation\\"
all_images_segmentation = images_list([seg_dir + classid + "\\" for classid in imagenetS50_ids_dict])

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(), # Converts to tensor
        transforms.ToDtype(torch.float32, scale=True)
    ]
)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# masks don't have channels

def segmentation_pipeline(images):
    transformed = []
    count = 0
    for img in all_images_segmentation:
        img_arr = np.array(img).astype(int)
        segmentation_id = torch.tensor(img_arr[:, :, 1] * 256 + img_arr[:, :, 0])
        mask = torch.eq(segmentation_id, (count // 10 + 1)).int().clone().detach()
        temp = torch.tensor(processor(torch.stack([mask, mask, mask]), do_rescale=False)['pixel_values'][0])
        values = temp.unique().tolist()
        # sometimes there are no 0's
        if values[0] != -1.0: values.insert(0, -1.0)
        if len(values) > 2: print("ERROR")
        temp.apply_(lambda val: values.index(val))
        transformed.append(temp.int()[0])
        count += 1

    return torch.stack(transformed)

ground_truth_masks = segmentation_pipeline(all_images_segmentation)
torch.save(ground_truth_masks, "ImageNet_VT/ground_truth_masks.pt")