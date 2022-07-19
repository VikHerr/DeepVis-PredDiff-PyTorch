

# number of test images
TESTS = 1
SAMPLE_STYLE = 'conditional'
WIN_SIZE = 10 
OVERLAPPING = 'stride' #'stride' 'None' 'full'
STRIDE      = 5 # only for stride
NUM_SAMPLES = 10
PADDING_SIZE = 2
# Imagenet 
#   'resnet50', 'mnv2', 'vgg16', 'alexnet'
# costum rsicd
#   'c_jakob'
MODEL = 'vgg16'

SOFTMAX = True 

SHOW = False  # show image before processing
IMG_IDX = 1

UPDATE = False

BATCH_SIZE = 64

#--- SELECT DATA SOURCE ---#
#*** IMAGENET VAL ***! THIS HAS TO BE SELECTED FOR GAUSSIAN PARAMETER OF IMAGENET!
DATASET_PATH = './data/val/'

#*** XAI RSICD TEST 
# DATASET_PATH = './data/RSICD/04_RSICD_XAI/'
#*** Airport rsicd
# DATASET_PATH = './data/RSICD/04_RSICD_XAI_Airport/'
#*** Beach rsicd
# DATASET_PATH = './data/RSICD/04_RSICD_XAI_Beach/'

#*** TEST INET    
IMAGE_PATH = './img/'
#IMAGE_PATH = './data/RSICD/04_RSICD_XAI/'

# imagenet 
RESULT_PATH = './results/'
# RSICD
# RESULT_PATH = './res_RSICD/'

# number of images for statistic paramter of conditional sampler
PARAM_DATASET_SIZE = 512 # IMAGENET
# PARAM_DATASET_SIZE = 100 # RSICD

# image net
CATEGORIE_PATH = './model/ilsvrc_2012_labels.txt'
# CATEGORIE_PATH = './model/rsicd_ood_lables.txt'