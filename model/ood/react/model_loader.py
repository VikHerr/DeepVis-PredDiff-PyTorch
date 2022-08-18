import torchvision as tv
import torch
import model.ood.react.resnet as rn



def htvc_resnet50(path, dset, num_of_classes=23, show=False):
    '''
    RESNET50 for multiple dataset
    '''

    model = rn.ResNet(rn.Bottleneck, [3,4,6,3],  num_of_classes)
    model.load_state_dict(torch.load(path))
    if show: 
        print('eval model: \n', model)

    dset_norms = {
        'mnist_fashion' : tv.transforms.Normalize((0.3201, 0.3182, 0.3629), (0.1804, 0.3569, 0.1131)),
        'cifar10'       : tv.transforms.Normalize((0.4881, 0.4660, 0.3994), (0.2380, 0.2322, 0.2413)),
        'sen12ms'        : tv.transforms.Normalize((0.1674, 0.1735, 0.2059), (0.1512, 0.1152, 0.1645)),
        'so2satlcz42'    : tv.transforms.Normalize((0.2380, 0.3153, 0.5004), (0.0798, 0.1843, 0.0666)),
        'xview2'         : tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'rsicd'          : tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        }

    print('using dset: ', dset)
    
    imagesize = 224

    preprocess = tv.transforms.Compose([
        tv.transforms.Resize((imagesize, imagesize)),
        tv.transforms.CenterCrop(imagesize),
        tv.transforms.ToTensor(),
        dset_norms[dset],
        ])

    # same crop for dispaly without norm
    display = tv.transforms.Compose([
        tv.transforms.Resize((imagesize, imagesize)),
        tv.transforms.CenterCrop(imagesize),
        tv.transforms.ToTensor(),
        ])

    return {'model' : model, 'name' : "jakob_{}".format(dset),'preprocess' : preprocess, 'display' : display, 'path' : path, 'ood' : None}


