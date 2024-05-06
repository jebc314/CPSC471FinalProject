Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe faithfulness.py ImageNet\limelasso_agg.pt ImageNet | Out-File Results/limelasso_agg_faithfulness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe randomization.py ImageNet\limelasso_agg.pt ImageNet | Out-File Results/limelasso_agg_randomization.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe sparseness.py ImageNet\limelasso_agg.pt ImageNet | Out-File Results/limelasso_agg_sparseness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe localization.py ImageNet\limelasso_agg.pt ImageNet | Out-File Results/limelasso_agg_localization.txt }

Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe faithfulness.py ImageNet\limelinearregression_agg.pt ImageNet | Out-File Results/limelinearregression_agg_faithfulness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe randomization.py ImageNet\limelinearregression_agg.pt ImageNet | Out-File Results/limelinearregression_agg_randomization.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe sparseness.py ImageNet\limelinearregression_agg.pt ImageNet | Out-File Results/limelinearregression_agg_sparseness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe localization.py ImageNet\limelinearregression_agg.pt ImageNet | Out-File Results/limelinearregression_agg_localization.txt }

Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe faithfulness.py ImageNet\limeridge_agg.pt ImageNet | Out-File Results/limeridge_agg_faithfulness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe randomization.py ImageNet\limeridge_agg.pt ImageNet | Out-File Results/limeridge_agg_randomization.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe sparseness.py ImageNet\limeridge_agg.pt ImageNet | Out-File Results/limeridge_agg_sparseness.txt }
Measure-Command { C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe localization.py ImageNet\limeridge_agg.pt ImageNet | Out-File Results/limeridge_agg_localization.txt }