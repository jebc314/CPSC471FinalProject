import torch
from torchvision.models import resnet18, ResNet18_Weights

import quantus

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

ground_truth_masks = torch.load(f"{sys.argv[2]}/ground_truth_masks.pt")
s_batch = ground_truth_masks
# From quantus:
# Make sure s_batch is of correct shape.
s_batch = s_batch.reshape(len(proper_data), 1, 224, 224)

relevance_mass_accuracy = quantus.RelevanceMassAccuracy()
localization_scores = relevance_mass_accuracy(
    model=model,
    x_batch=proper_data.cpu().numpy(),
    y_batch=preds.cpu().numpy(),
    a_batch=agg.sum(dim = 1, keepdim = True).detach().cpu().numpy(), # because quantus sums each pixel's attribution for some reason
    s_batch=s_batch.cpu().numpy(),
    device=device,
    channel_first=True
)
np.save(f"Results/localization_RMA_scores{time.time()}.npy", np.array(localization_scores))

print("Average")
print(sum(localization_scores)/len(localization_scores))
print("Uncertainty")
print(np.std(localization_scores) / np.sqrt(len(localization_scores)))