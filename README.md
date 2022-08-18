# Pytorch Visualizing Deep Neural Network Decisions

Pytorch implemetation of the paper:

"VISUALIZING DEEP NEURAL NETWORK DECISIONS:
PREDICTION DIFFERENCE ANALYSIS"

This repo implements the methods using the Pytorch framework. 
The original implementation by the Authors can be found here:

link: https://github.com/lmzintgraf/DeepVis-PredDiff

The code is mostly adapted from the linked git repo.
Added classification functions that replace the caffe framework with PyTorch and a dataloader 
Also added option for variable stride of PDA to reduce run time
___
## System Requirements / Tested with:
    python 3.6.9
    numpy 1.18.5
    torch 1.9.0+cu111
    matplotlib 3.3.4
    PIL 8.1.0
    scipy 1.5.4
    skimage 0.17.2
___
## Usage
Models trained on the IMAGENET ILSVRC2012 dataset are loaded from torchvison and can be used out of the box.
Some example images are available in:

    <PROJECT_ROOTDIR>/data/ilsvrc_2012_test 

Check the config.py for settings and set the dataset to 'ilscvrv_2012_test'. Choose Imagenet model from config. For processing run:

    python3 main.py

Use the '-v' option to show results with same settings or the '-c' option to convert heatmaps to uint8 (with same settings, after processing)

    python3 main.py -v
    python3 main.py -c

To reduce the runtime, change the OVERLAP paramter in config.py. This will also reduce the quality of the output HEATMAP
___
## Adding Datasets
Add the data folder to:

    <PROJECT_ROOTDIR>/data/<DATASET_NAME>

The DATASET_NAME has to match the name in the config file

___
## Using Costum Models
add the .pth file to the folder of the model 

    <PROJECT_ROOTDIR>/model/ood/<METHOD>/

with available METHODs:
- react: (ResNet50)
- isomax: (ResNet50 + isomax)

For resnet50 change the name of the .pth file to:

    resnet50_NAME_DATSET.pth

For resnet50 + regression models change the name of the .pth file to:

    REGRESSION_resnet50_NAME_DATSET.pth

The parameter have to match the settings from the config file



