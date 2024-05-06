import torch
from torchvision.models import resnet18, ResNet18_Weights
from captum.attr import IntegratedGradients, Saliency, GradientShap, GuidedBackprop, Deconvolution, InputXGradient, Lime, Occlusion, ShapleyValueSampling, FeatureAblation, KernelShap, NoiseTunnel

import quantus

from skimage.segmentation import quickshift
from EnsembleXAI.Ensemble import normEnsembleXAI
from EnsembleXAI.Normalization import mean_var_normalize

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

# attributions
integrated_gradients = IntegratedGradients(model)
saliency = Saliency(model)
gradient_shap = GradientShap(model)
guided_backprop = GuidedBackprop(model)
deconvolution = Deconvolution(model)
input_x_gradient = InputXGradient(model)
lime = Lime(model)
occulsion = Occlusion(model)
shapley_value_sampling = ShapleyValueSampling(model)
feature_ablation = FeatureAblation(model)
kernel_shap = KernelShap(model)
noise_tunnel = NoiseTunnel(integrated_gradients)

def explain(model, inputs, targets, **kwargs):
    masks = torch.stack([torch.tensor(quickshift(image, kernel_size=4, max_dist=200, ratio=0.2, channel_axis=0)) for image in inputs]).cuda()
    inputs = torch.tensor(inputs).cuda()
    targets = torch.tensor(targets).cuda()
    
    # attributions
    attributions = [
        integrated_gradients.attribute(inputs, target = targets),
        saliency.attribute(inputs, target = targets),
        gradient_shap.attribute(inputs, torch.zeros_like(inputs), target=targets),
        guided_backprop.attribute(inputs, target = targets),
        deconvolution.attribute(inputs, target = targets),
        input_x_gradient.attribute(inputs, target = targets),
        lime.attribute(inputs, target=targets, feature_mask=masks),
        occulsion.attribute(inputs, (3, 15, 15), target=targets, strides = (3, 8, 8)),
        shapley_value_sampling.attribute(inputs, target=targets, feature_mask=masks),
        feature_ablation.attribute(inputs, target=targets, feature_mask=masks),
        kernel_shap.attribute(inputs, target=targets, feature_mask=masks),
        noise_tunnel.attribute(inputs, target=targets),
    ]

    # normalized
    normalized_attributions = [mean_var_normalize(attribution) for attribution in attributions]
    
    explanations = torch.stack(normalized_attributions, dim=1)
    agg = normEnsembleXAI(explanations.detach(), aggregating_func='avg')
    return agg.cpu().numpy()

#randomization
random_logit = quantus.RandomLogit(
    num_classes=(10 if sys.argv[2] == "MNIST" else 1000),
    similarity_func=quantus.similarity_func.ssim,
    display_progressbar=True
)

randomization_scores = random_logit(
    model=model,
    x_batch=proper_data.cpu().numpy(),
    y_batch=preds.cpu().numpy(),
    a_batch=agg.detach().cpu().numpy(),
    device=device,
    explain_func=explain, 
    explain_func_kwargs={},
)

np.save(f"Results/randomization_scores{time.time()}.npy", np.array(randomization_scores))

print("Average")
print(sum(randomization_scores)/len(randomization_scores))
print("Uncertainty")
print(np.std(randomization_scores) / np.sqrt(len(randomization_scores)))