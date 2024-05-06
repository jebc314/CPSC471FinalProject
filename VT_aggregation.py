import torch
from EnsembleXAI.Ensemble import normEnsembleXAI
from EnsembleXAI.Normalization import mean_var_normalize

explainers = [
    'attributions_ig',
    'attributions_s',
    'attributions_gs',
    'attributions_gb',
    'attributions_d',
    'attributions_ixg',
    'attributions_l',
    'attributions_o',
    'attributions_svs',
    'attributions_fa',
    'attributions_ks',
    'attributions_nt'
]

attributions = {}

for explainer in explainers:
    attribution = None
    for i in range(50):
        if attribution is not None:
            attribution = torch.cat((attribution, torch.load(f'ImageNet_VT/{explainer}-{i}.pt')))
        else:
            attribution = torch.load(f'ImageNet_VT/{explainer}-{i}.pt')
    attributions[explainer] = attribution
    torch.save(attribution, f"ImageNet_VT/{explainer}.pt")

normalized_attributions = {attr: mean_var_normalize(attributions[attr]) for attr in attributions}

explanations = torch.stack([normalized_attributions[attr] for attr in normalized_attributions], dim=1)

agg = normEnsembleXAI(explanations.detach(), aggregating_func='avg')

torch.save(agg, "ImageNet_VT/agg.pt")

# Also aggregate the preds and proper data
preds = None
proper_data = None
for i in range(50):
    if preds is not None:
        preds = torch.cat((preds, torch.load(f'ImageNet_VT/preds-{i}.pt')))
    else:
        preds = torch.load(f'ImageNet_VT/preds-{i}.pt')
    if proper_data is not None:
        proper_data = torch.cat((proper_data, torch.load(f'ImageNet_VT/proper_data{i}.pt')))
    else:
        proper_data = torch.load(f'ImageNet_VT/proper_data{i}.pt')

torch.save(preds, "ImageNet_VT/preds.pt")
torch.save(proper_data, "ImageNet_VT/proper_data.pt")