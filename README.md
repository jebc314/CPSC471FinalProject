# CPSC471FinalProject
My CPSC 471 final project code for reproducibility. 

## Setup

Refer to the Conda environment configuration file environment.yml for the libraries required for running experiments.

In order to process_data.ipynb, you need to have the [ImageNet S-50 annotation dataset](https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip) downloaded into "input\ImageNetS50". You also need to download the [ILSVRC2012_img_train\<CLS>.tar files](https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) into "input\images". The output is the "input\images\<CLS>" will have the images correponding to the annotations in "input\ImageNetS50".

For recreating the paper's tests on their aggregation functions, you need to create an ImageNet folder. Then, running paper_replication.ipynb will produce "ImageNet/proper_data.pt", "ImageNet/preds.pt", "ImageNet/attributions_*.pt" for each of the attributions, and "ImageNet/agg.pt", which is the file with the aggregated explanations. To get the metric test results, you need to first create the ground truth masks using "ground_truth_mask.py". Then, create a "Results" folder and run faithfullness.py, randomization.py, robustness_file.py, sparseness.py, and localization.py to produce the corresponding metric scores.

Using the command "python <FILE> ImageNet/<AGGREGATED EXPLANATION>.pt ImageNet" will produce the results. Refer to run_tests.ps1 for an example PowerShell script that produces metric results.

## Aggregation Functions Experiments

For creating the aggregated explanations with the new aggregation functions refer to the second multiline comment in aggregation.py. Running python aggregation.py outputs the aggregated explanations in the "ImageNet" folder.

The metric_tests.ipynb file will produce the weighted and threshold aggregated explanations.

Use the same process as in setup to get the metric test results.

## Pre-Aggregation Modifications

For the subset of explanations (gradient-based versus perturbation based), refer to the first multiline comment in aggregation.py.

In terms of the hyperparameter tuning experiments, the hyperparameter_tuning.py will produce the optimal hyperparameters for each the explainers. Note you have to create a "HyperparameterSearch" folder before running.

Run hyperparameter_analysis.ipynb to get the aggregated explanation. The caveaut is that the individual attribution names have to be changed to the respective optimal one from runnning hyperparameter because they are timestamped.

For getting the attributions with the base model LIME uses changed, run improved_LIME.ipynb and the aggregated explanations will be saved in the "ImageNet" folder.

Use the same process as in setup to get the metric test results.

## Scope Expansion

For the CIFAR10 and MNIST datasets, the respective notebooks paper_replication_cifar10.ipynb and paper_replication_MNIST.ipynb up till the section Ground Truth Masks can be run directly. For ground truth masks, these sections had more manual changes because of the inconsistent results, which can make it more challenging to run. Refer to "CIFAR10/ground_truth_masks.pt" and "MNIST/ground_truth_masks.pt" for ground truth masks for these two datasets. They can be easily verified by plotting using Matplotlib's imshow. Refer to ground_truth_mask_tool.py for the tool that I used to make some masks manually. This tool needs the folder "CIFAR10/data/" with the images to mask and the folder "CIFAR10/masks/" to store the masks. The "N" button moves to the next picture, and the "S" button saves the mask.

For running the aggregation process on a vision trasformer, you need to create the folder "ImageNet_VT". The VT_generate_preds.py, VT_generate_expanations.py, and "VT_aggregation.py" will generate the aggregated explanations. The powershell run_VT_pars.ps1 has an example script to generate explanations using a subset of the explainers. The file ground_truth_mask_VT.py will generate the ground truth masks.

Use the same process as in setup to get the metric test results. Just make sure to change the folder name to the respective tests, like ImageNet to MNIST or to ImageNet_VT.

The files randomization_eMPRT.py and localization_RMA.py will evalute an explanation according to the respective metrics. Refer to run_tests_metric.ps1 for an example run evaluating some aggregated explanations.