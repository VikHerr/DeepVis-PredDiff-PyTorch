

# number of test images
TESTS = 4
IMG_IDX = 8

SAMPLE_STYLE = 'conditional'
WIN_SIZE = 20 
OVERLAPPING = 'stride' #'stride' 'None' 'full'
STRIDE      = 5 # only for stride
NUM_SAMPLES = 10
PADDING_SIZE = 2
# Imagenet 
#   'resnet50', 'mnv2', 'vgg16', 'alexnet'
# costum rsicd
#   'c_jakob' 'c_conrad
MODEL = 'c_jakob'


SOFTMAX = True
OOD     = False

SHOW = False  # show image before and after processing

UPDATE = False

BATCH_SIZE = 64

#-------------------
# LAPLACEN CORRECTION PDA IMAGENET
TRAINSIZE = 100000
CLASSES   = 1000

# RSICD (does not change anything!)
# TRAINSIZE = 10921


# DATASETS:
# 'so2satlcz42' (10), 'xview2'(7), 'rsicd'(23), 'sen12ms'(9), 'mnist_fashion'(7), 'cifar10'(6) ', 'ilsvrc_2012'(1000)
DSET = 'rsicd'
CLASSES   = 23

#'jakob', 'rabby', 'conrad', 'samuel'
NAME = '_jakob_' # _conrad_' # ''
# 'regression_' or '' for no regression 
REGRESSION = '' # 'regression_' 
# comment in to not use regression models!!
# REGRESSION = ''
if REGRESSION != '':
    CLASSES = 1

# image net
# CATEGORIE_PATH = './model/ilsvrc_2012_labels.txt'
CATEGORIE_PATH = './model/' + DSET + '_labels.txt'

if OOD or REGRESSION != '':
    # no real lables, dummy for ood score
    CATEGORIE_PATH = './model/ood_lables.txt'


#--- SELECT DATA SOURCE ---#
# This path is used to fit gauissan model so there 
# should be a high number of samples in this folder 
#*** IMAGENET VAL ***! THIS HAS TO BE SELECTED FOR GAUSSIAN PARAMETER OF IMAGENET!
# DATASET_PATH = './data/val/'
#*** XAI RSICD TEST 
DATASET_PATH = './data/' + DSET + '/' 


#*** TEST INET    
IMAGE_PATH = DATASET_PATH

# only use for handpicked examples!
#IMAGE_PATH = './img/'

RESULT_PATH = './res_' + REGRESSION +  DSET + '{}'.format(NAME) +  str(int(OOD)) +  '/'

# number of images for statistic paramter of conditional sampler
# PARAM_DATASET_SIZE = 512 # IMAGENET
PARAM_DATASET_SIZE = 100 # RSICD

