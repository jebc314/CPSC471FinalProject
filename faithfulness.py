import torch
from torchvision.models import resnet18, ResNet18_Weights

import quantus
from quantus.helpers.utils import calculate_auc

import numpy as np

import sys
import time

from transformers import ViTForImageClassification

# For the vision transformer
class Module(torch.nn.Module):
    # Initialize the parameter 
    def __init__(self, model): 
        super(Module, self).__init__() 
        self.model = model
    # Forward pass 
    def forward(self, inputs): 
        outputs = self.model(inputs)
        return outputs["logits"]
    def apply(self, func):
        for m in self.model.modules():
            func(m)

# Load model
if sys.argv[2] == "ImageNet_VT":
    # sets up model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    temp = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    temp.eval()
    model = Module(temp).eval()
elif sys.argv[2] == "MNIST":
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 10, bias=True)
    model.load_state_dict(torch.load(r"MNIST\model.pt"))
    model.eval()
    model = model.cuda()
else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    model = model.cuda()

# Load tensors
proper_data = torch.load(f"{sys.argv[2]}/proper_data.pt").cuda()
preds = torch.load(f"{sys.argv[2]}/preds.pt").cuda()
proper_masks = torch.load(f"{sys.argv[2]}/proper_masks.pt").cuda()

device = torch.device("cuda")
agg = torch.load(f"{sys.argv[1]}")

pixel_flipping = quantus.PixelFlipping(
    features_in_step=224,
    perturb_baseline="black",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    # return_auc_per_sample=True,
    display_progressbar=True
)
    
# Call the metric instance to produce scores.
faithfulness_scores = pixel_flipping(
    model=model,
    x_batch=proper_data.cpu().numpy(),
    y_batch=preds.cpu().numpy(),
    a_batch=agg.detach().cpu().numpy(),
    device=device,
)

np.save(f"Results/faithfulness_scores{time.time()}.npy", np.array(faithfulness_scores))

f = calculate_auc(faithfulness_scores, dx=1 / len(faithfulness_scores[0]))
print("Average")
print(sum(f) / len(f))
print("Uncertainty")
print(np.std(f) / np.sqrt(len(f)))