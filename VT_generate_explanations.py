from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from captum.attr import IntegratedGradients, Saliency, GradientShap, GuidedBackprop, Deconvolution, InputXGradient, Lime, Occlusion, ShapleyValueSampling, FeatureAblation, KernelShap, NoiseTunnel
import sys

# sets up model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
model.eval();

# load data
proper_data = torch.load(f"ImageNet_VT/proper_data{sys.argv[1]}.pt").cuda()
preds = torch.load(f"ImageNet_VT/preds-{sys.argv[1]}.pt").cuda()
proper_masks = torch.load("ImageNet/proper_masks.pt")[10*int(sys.argv[1]):10*(int(sys.argv[1]) + 1)].cuda()

# explanation generation

def forward_func(inputs):
    outputs = model(inputs)
    return outputs["logits"]

def get_attributions(explainer, num_batches = 50):
    attributions = None
    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions is None:
            attributions = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice])
        else:
            temp = explainer.attribute(proper_data[batch_slice], target=preds[batch_slice])
            attributions = torch.cat((attributions, temp), dim = 0)
    return attributions


# To handle the fact that model is not an nn.Module object
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

# For the custom ones
num_batches = 1

if sys.argv[2] == "ig":
    integrated_gradients = IntegratedGradients(forward_func)
    attributions_ig = get_attributions(integrated_gradients, num_batches=10)
    torch.save(attributions_ig, f"ImageNet_VT/attributions_ig-{sys.argv[1]}.pt")
elif sys.argv[2] == "s":
    saliency = Saliency(forward_func)
    attributions_s = get_attributions(saliency, num_batches=10)
    torch.save(attributions_s, f"ImageNet_VT/attributions_s-{sys.argv[1]}.pt")
elif sys.argv[2] == "gb":
    guided_backprop = GuidedBackprop(Module(model))
    attributions_gb = get_attributions(guided_backprop, num_batches=10)
    torch.save(attributions_gb, f"ImageNet_VT/attributions_gb-{sys.argv[1]}.pt")
elif sys.argv[2] == "d":
    deconvolution = Deconvolution(Module(model))
    attributions_d = get_attributions(deconvolution, num_batches = 10)
    torch.save(attributions_d, f"ImageNet_VT/attributions_d-{sys.argv[1]}.pt")
elif sys.argv[2] == "ixg":
    input_x_gradient = InputXGradient(forward_func)
    attributions_ixg = get_attributions(input_x_gradient, num_batches = 10)
    torch.save(attributions_ixg, f"ImageNet_VT/attributions_ixg-{sys.argv[1]}.pt")
elif sys.argv[2] == "nt":
    integrated_gradients = IntegratedGradients(forward_func)
    noise_tunnel = NoiseTunnel(integrated_gradients) # base on EnsembleXAI
    attributions_nt = get_attributions(noise_tunnel, num_batches = 10)
    torch.save(attributions_nt, f"ImageNet_VT/attributions_nt-{sys.argv[1]}.pt")
elif sys.argv[2] == "gs":
    gradient_shap = GradientShap(forward_func)
    attributions_gs = None

    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_gs is None:
            attributions_gs = gradient_shap.attribute(proper_data[batch_slice].cuda(), torch.zeros_like(proper_data[0:1]), target=preds[batch_slice].cuda())
        else:
            temp = gradient_shap.attribute(proper_data[batch_slice].cuda(), torch.zeros_like(proper_data[0:1]), target=preds[batch_slice].cuda())
            attributions_gs = torch.cat((attributions_gs, temp), dim = 0)
    torch.save(attributions_gs, f"ImageNet_VT/attributions_gs-{sys.argv[1]}.pt")
elif sys.argv[2] == "l":
    # Need feature mask
    lime = Lime(forward_func)
    attributions_l = None

    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_l is None:
            attributions_l = lime.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
        else:
            temp = lime.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
            attributions_l = torch.cat((attributions_l, temp), dim = 0)
    torch.save(attributions_l, f"ImageNet_VT/attributions_l-{sys.argv[1]}.pt")
elif sys.argv[2] == "o":
    occulsion = Occlusion(forward_func)
    attributions_o = None

    # Using sliding window size (3, 15, 15) and strides = (3, 8, 8) as used in EnsembleXAI
    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_o is None:
            attributions_o = occulsion.attribute(proper_data[batch_slice].cuda(), (3, 15, 15), target=preds[batch_slice].cuda(), strides = (3, 8, 8))
        else:
            temp = occulsion.attribute(proper_data[batch_slice].cuda(), (3, 15, 15), target=preds[batch_slice].cuda(), strides = (3, 8, 8))
            attributions_o = torch.cat((attributions_o, temp), dim = 0)
    torch.save(attributions_o, f"ImageNet_VT/attributions_o-{sys.argv[1]}.pt")
elif sys.argv[2] == "svs":
    # Need feature mask
    shapley_value_sampling = ShapleyValueSampling(forward_func)
    attributions_svs = None

    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_svs is None:
            attributions_svs = shapley_value_sampling.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
        else:
            temp = shapley_value_sampling.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
            attributions_svs = torch.cat((attributions_svs, temp), dim = 0)
    torch.save(attributions_svs, f"ImageNet_VT/attributions_svs-{sys.argv[1]}.pt")
elif sys.argv[2] == "fa":
    # Need feature mask
    feature_ablation = FeatureAblation(forward_func)
    attributions_fa = None

    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_fa is None:
            attributions_fa = feature_ablation.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
        else:
            temp = feature_ablation.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
            attributions_fa = torch.cat((attributions_fa, temp), dim = 0)
    torch.save(attributions_fa, f"ImageNet_VT/attributions_fa-{sys.argv[1]}.pt")
elif sys.argv[2] == "ks":
    # Need feature mask
    kernel_shap = KernelShap(forward_func)
    attributions_ks = None

    for i in range(num_batches):
        batch_slice = slice(i * len(proper_data) // num_batches, (i + 1) * len(proper_data) // num_batches)
        if attributions_ks is None:
            attributions_ks = kernel_shap.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
        else:
            temp = kernel_shap.attribute(proper_data[batch_slice], target=preds[batch_slice], feature_mask=proper_masks[batch_slice])
            attributions_ks = torch.cat((attributions_ks, temp), dim = 0)
    torch.save(attributions_ks, f"ImageNet_VT/attributions_ks-{sys.argv[1]}.pt")