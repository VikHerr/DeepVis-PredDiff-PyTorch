## Pytorch Visualizing Deep Neural Network Decisions (Work in progress!)

Pytorch implemetation of the paper:

"VISUALIZING DEEP NEURAL NETWORK DECISIONS:
PREDICTION DIFFERENCE ANALYSIS"

This repo implements the methods using the Pytorch framework. 
The original implementation by the Authors can be found here:

link: https://github.com/lmzintgraf/DeepVis-PredDiff

The code is mostly adapted from the linked git repo.
Added classification functions that replace the caffe framework with PyTorch and a dataloader 
Also added option for variable stride of PDA to reduce run time

So far:
Tested with MobilenetV2

For now: Only use Conditional sampling method


TODO:
- add howto to README file
- add other models
- Test with odd methods
