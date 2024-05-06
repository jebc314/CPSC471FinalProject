# %% [markdown]
# Reference: https://github.com/Hryniewska/EnsembleXAI/blob/main/notebooks/Imagenet_tests_and_results.ipynb

# %%
import torch
from torchvision.models import resnet18, ResNet18_Weights
from captum.attr import IntegratedGradients, Saliency, GradientShap, GuidedBackprop, Deconvolution, InputXGradient, Lime, Occlusion, ShapleyValueSampling, FeatureAblation, KernelShap, NoiseTunnel
from EnsembleXAI.Metrics import accordance_precision, accordance_recall
import pyhopper
import time

# %% [markdown]
# ### Model

# %%
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
model_cuda = model.cuda()

# %% [markdown]
# ### Attributions (default parameters)

# %%
# Load tensors
proper_data = torch.load("ImageNet/proper_data.pt").cuda()
preds = torch.load("ImageNet/preds.pt").cuda()
proper_masks = torch.load("ImageNet/proper_masks.pt").cuda()
ground_truth_masks = torch.load("ImageNet/ground_truth_masks.pt").cuda()

# %%
# Custom F1_score so that when precision and recall = 0 (F1 nan) that is just interpretted as zero 
# (throwing away the consideration of that point or making really bad -> just bad)
def F1_score(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> float:
    acc_recall = accordance_recall(explanations, masks, threshold=threshold)
    acc_prec = accordance_precision(explanations, masks, threshold=threshold)
    values = 2 * (acc_recall * acc_prec) / (acc_recall + acc_prec)
    values[values != values] = 0
    value = torch.sum(values) / values.shape[0]
    return value.item()

# %%
def get_attributions(explainer, num_batches = 50, **kwargs):
    attributions = None
    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions is None:
            attributions = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice], **kwargs)
        else:
            temp = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice], **kwargs)
            attributions = torch.cat((attributions, temp), dim = 0)
    return attributions

# %%
explainers = {
    'integrated_gradients': IntegratedGradients(model_cuda),
    'saliency': Saliency(model_cuda),
    'gradient_shap': GradientShap(model_cuda),
    'guided_backprop': GuidedBackprop(model_cuda),
    'deconvolution': Deconvolution(model_cuda),
    'input_x_gradient': InputXGradient(model_cuda),
    'lime': Lime(model_cuda),
    'occulsion': Occlusion(model_cuda),
    'shapley_value_sampling': ShapleyValueSampling(model_cuda),
    'feature_ablation': FeatureAblation(model_cuda),
    'kernel_shap': KernelShap(model_cuda),
    'noise_tunnel': NoiseTunnel(IntegratedGradients(model_cuda)) # base on EnsembleXAI
}

# %%
def test_attribute(attribute_type, params: dict) -> float:
    if "sliding_window_shapes" in params:
        params["sliding_window_shapes"] = (3, params["sliding_window_shapes"], params["sliding_window_shapes"])
    if "strides" in params:
        params["strides"] = (3, params["strides"], params["strides"])
    attributions = None
    num_batches = 500
    explainer = explainers[attribute_type]
    if attribute_type == "lime" or attribute_type == "shapley_value_sampling" or attribute_type == "feature_ablation" or attribute_type == "kernel_shap":
        for i in range(num_batches):
            batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
            if attributions is None:
                attributions = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice], **params)
            else:
                temp = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice], **params)
                attributions = torch.cat((attributions, temp), dim = 0)
    else:
        attributions = get_attributions(explainer, num_batches = num_batches, **params)

    f1_score = F1_score(attributions, ground_truth_masks)

    torch.save(attributions, f"HyperparameterSearch/attributions_{attribute_type}_{f1_score:.2f}_{time.time()}.pt")

    return F1_score(attributions, ground_truth_masks)

# %%
from PIL import Image
from torchvision.models import ResNet18_Weights
import numpy as np
np.random.seed(42)
resnet_transform = ResNet18_Weights.DEFAULT.transforms()
baselines = [
    torch.zeros_like(proper_data[0]), # 0
    resnet_transform(Image.new('RGB',(224,224),"rgb(128,128,128)")), # gray
    resnet_transform(Image.new('RGB',(224,224),"rgb(255,255,255)")), # white
    resnet_transform(Image.new('RGB',(224,224),"rgb(0,0,0)")), # black
    resnet_transform(Image.fromarray((np.random.randint(low=0,high=256,size=128*128*3, dtype=np.uint8)).reshape(128,128,3),'RGB')) # static
]

# %%
baselines = [torch.stack([baseline]).cuda() for baseline in baselines]

# %%
# integrated_gradients
search_parameters = {
    "baselines": pyhopper.choice(baselines),
    "n_steps": pyhopper.int(50,100),
    "method": pyhopper.choice(["riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"])
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("integrated_gradients", params), 
    direction="max", 
    runtime="1h", 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/integrated_gradients.ckpt"
)
print(f"integrated_gradients: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# saliency
search_parameters = {
    "abs": pyhopper.choice([True, False])
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("saliency", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/saliency.ckpt"
)
print(f"saliency: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# gradient_shap
search_parameters = {
    "baselines": pyhopper.choice(baselines),
    "n_samples": pyhopper.int(5,15),
    "stdevs": pyhopper.float(0, 1)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("gradient_shap", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/gradient_shap.ckpt"
)
print(f"gradient_shap: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# guided_backprop -> no hyperparameters to tune
# deconvolution -> no hyperparameters to tune
# input_x_gradient -> no hyperparameters

# %%
# lime
search_parameters = {
    "n_samples": pyhopper.int(50,100)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("lime", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/lime.ckpt"
)
print(f"lime: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# occulsion sliding window size (3, 15, 15) and strides = (3, 8, 8)
search_parameters = {
    "sliding_window_shapes": pyhopper.int(10, 20),
    "strides": pyhopper.int(5, 15),
    "baselines": pyhopper.choice(baselines)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("occulsion", params), 
    direction="max", 
    runtime="2h", 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/occulsion.ckpt"
)
print(f"occulsion: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# shapley_value_sampling
search_parameters = {
    "baselines": pyhopper.choice(baselines),
    "n_samples": pyhopper.int(25, 50)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("shapley_value_sampling", params), 
    direction="max", 
    runtime="3h", 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/shapley_value_sampling.ckpt"
)
print(f"shapley_value_sampling: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# feature_ablation
search_parameters = {
    "baselines": pyhopper.choice(baselines)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("feature_ablation", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/feature_ablation.ckpt"
)
print(f"feature_ablation: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# kernel_shap
search_parameters = {
    "baselines": pyhopper.choice(baselines),
    "n_samples": pyhopper.int(50,100)
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("kernel_shap", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/kernel_shap.ckpt"
)
print(f"kernel_shap: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%
# noise_tunnel
search_parameters = {
    "nt_type": pyhopper.choice(["smoothgrad","smoothgrad_sq","vargrad"]),
    "nt_samples": pyhopper.int(5, 15),
    "stdevs": pyhopper.float(0, 2),
    "baselines": pyhopper.choice(baselines),
    "n_steps": pyhopper.int(50,100),
    "method": pyhopper.choice(["riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"])
}
search = pyhopper.Search(
    search_parameters
)
best_params = search.run(
    lambda params: test_attribute("noise_tunnel", params), 
    direction="max", 
    steps=10, 
    n_jobs="per-gpu",
    checkpoint_path="HyperparameterSearch/noise_tunnel.ckpt"
)
print(f"noise_tunnel: {best_params}")
for key in search_parameters:
    print(f"{key}: {search.history[key]}")
print(f"{search.history.fs}")

# %%



