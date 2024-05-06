import torch
from EnsembleXAI.Ensemble import normEnsembleXAI
from EnsembleXAI.Normalization import mean_var_normalize

attributions = {
    'attributions_ig': torch.load('ImageNet/attributions_ig.pt'),
    'attributions_s': torch.load('ImageNet/attributions_s.pt'),
    'attributions_gs': torch.load('ImageNet/attributions_gs.pt'),
    'attributions_gb': torch.load('ImageNet/attributions_gb.pt'),
    'attributions_d': torch.load('ImageNet/attributions_d.pt'),
    'attributions_ixg': torch.load('ImageNet/attributions_ixg.pt'),
    'attributions_l': torch.load('ImageNet/attributions_l.pt'),
    'attributions_o': torch.load('ImageNet/attributions_o.pt'),
    'attributions_svs': torch.load('ImageNet/attributions_svs.pt'),
    'attributions_fa': torch.load('ImageNet/attributions_fa.pt'),
    'attributions_ks': torch.load('ImageNet/attributions_ks.pt'),
    'attributions_nt': torch.load('ImageNet/attributions_nt.pt'),
}
# For other subsets replace the above variable declaration with the respective section:
'''
Gradient
attributions = {
    'attributions_ig': torch.load('ImageNet/attributions_ig.pt'),
    'attributions_s': torch.load('ImageNet/attributions_s.pt'),
    'attributions_gs': torch.load('ImageNet/attributions_gs.pt'),
    'attributions_gb': torch.load('ImageNet/attributions_gb.pt'),
    'attributions_d': torch.load('ImageNet/attributions_d.pt'),
    'attributions_ixg': torch.load('ImageNet/attributions_ixg.pt'),
}

Perturbation
attributions = {
    'attributions_l': torch.load('ImageNet/attributions_l.pt'),
    'attributions_o': torch.load('ImageNet/attributions_o.pt'),
    'attributions_svs': torch.load('ImageNet/attributions_svs.pt'),
    'attributions_fa': torch.load('ImageNet/attributions_fa.pt'),
    'attributions_ks': torch.load('ImageNet/attributions_ks.pt'),
}
'''

normalized_attributions = {attr: mean_var_normalize(attributions[attr]) for attr in attributions}

def median(tensor):
    return torch.quantile(tensor, q = 0.5, dim = 0)

explanations = torch.stack([normalized_attributions[attr] for attr in normalized_attributions], dim=1)
agg = normEnsembleXAI(explanations.detach(), aggregating_func='avg')
# For other aggregation methods, replace the above two lines with the respective section
'''
Median
explanations = torch.stack([normalized_attributions[attr] for attr in normalized_attributions], dim=1)
agg = normEnsembleXAI(explanations.detach(), aggregating_func=median)

Average absolute value
explanations = torch.stack([torch.abs(normalized_attributions[attr]) for attr in normalized_attributions], dim=1)
agg = normEnsembleXAI(explanations.detach(), aggregating_func='avg')

Range
explanations = torch.stack([normalized_attributions[attr] for attr in normalized_attributions], dim=1)
agg_min = normEnsembleXAI(explanations.detach(), aggregating_func='min')
agg_max = normEnsembleXAI(explanations.detach(), aggregating_func='max')
agg = agg_max - agg_min

Minimum absolute value
explanations = torch.stack([torch.abs(normalized_attributions[attr]) for attr in normalized_attributions], dim=1)
agg = normEnsembleXAI(explanations.detach(), aggregating_func='min')
'''

torch.save(agg, "ImageNet/<AGGREGATION NAME>_agg.pt")