from statistics import mode
import numpy as np
import torch 
import torchvision as tv
from PIL import Image
import time
import dataloader as udl 
import classifiers as ucls
import visualizer as uv
import os
import sampler as sampling
from config import *
import matplotlib.pyplot as plt

from pred_diff_analyser import PredDiffAnalyser

# ood model
import model.ood.react.resnet as rn


def infoResNet(show=False,mVersion='resnet50'):
    '''
    ResNet model info
    '''

    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    if mVersion == 'resnet50':
        model = tv.models.resnet50(pretrained=True)
    elif mVersion == 'resnet18':
        model = tv.models.resnet18(pretrained=True)
    else:
        assert False, 'invaled resnet version'

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : mVersion, 'preprocess' : preprocess, 'display' : display, 'path' : ''}

def infoMobileNetV2(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.mobilenet_v2(pretrained=True)
    if show: 
        print('eval model: \n', model)

    # model.eval()

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "moblienet_v2",'preprocess' : preprocess, 'display' : display, 'path' : ''}

def infoAlexNet(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.alexnet(pretrained=True, progress=True)

    if show: 
        print('eval model: \n', model)

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "alexnet",'preprocess' : preprocess, 'display' : display, 'path' : ''}

def infoVgg16(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.vgg16(pretrained=True, progress=True)

    if show: 
        print('eval model: \n', model)

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "vgg16",'preprocess' : preprocess, 'display' : display, 'path' : ''}


def infoCostumRSICD(path, show=False):
    '''
    RSIDCD model info RESNET50
    '''

    model = rn.ResNet(rn.Bottleneck, [3,4,6,3], 23)
    model.load_state_dict(torch.load(path))
    if show: 
        print('eval model: \n', model)

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "costum",'preprocess' : preprocess, 'display' : display, 'path' : path}



def test_prob(classifier, x):
    '''
    test predict function 
    '''
    top_prob, top_catid = classifier.getTopCat(x[0:1])
    
    cat_prob = classifier.predict(x) 

    print(np.max(cat_prob), np.min(cat_prob))

def viewImage(model):

    dataLoader = udl.DataLoader(model=model, path='./img/')
    #dataLoader = udl.DataLoader(model=model)# , path='./img/')

    test_size = TESTS
    show      = True
    X_test,X_test_im, X_filenames = dataLoader.get_imagenet_data(s_idx=IMG_IDX, b_size=test_size, set_size=1)

    classifier = ucls.Classifier(model, softmax=SOFTMAX)

    res = classifier.predict(X_test)
    print('RES SHAPE: ', res.shape)
    print(res)
    # classifier.softmax = False   
    # test_prob(classifier, X_test)
    # classifier.softmax = True
    # test_prob(classifier, X_test)

    # for test_idx in range(test_size):

    #     if show:
    #         plt.imshow(X_test_im[test_idx])
    #         plt.show()

    #     top_prob, top_catid = classifier.getTopCat(X_test[test_idx:test_idx+1])
    #     y_pred_label = classifier.categories[top_catid]
    #     print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))


def experiment(model):

    dataLoader = udl.DataLoader(model=model, path='./img/')

    test_size = TESTS
    show      = SHOW
    softmax = SOFTMAX
    X_test,X_test_im, X_filenames = dataLoader.get_imagenet_data(s_idx=IMG_IDX, b_size=test_size, set_size=8)

    path_results = './results/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)  

    classifier = ucls.Classifier(model, softmax=softmax)
    # Test: get prob of class from 
    #test_prob(classifier,X_test)


    for test_idx in range(test_size):

        x_test = X_test[test_idx]
        x_test_im = X_test_im[test_idx]

        top_prob, top_catid = classifier.getTopCat(x_test[np.newaxis])

        y_pred_label = classifier.categories[top_catid]
        print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))



        # show image
        if show:
            plt.imshow(x_test_im)
            plt.show()

        start_time = time.time()

        if SAMPLE_STYLE == 'conditional':
            save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}.jpg'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE,NUM_SAMPLES,PADDING_SIZE,classifier.name)
            sampler = sampling.ConditionalSampler(win_size=WIN_SIZE, padding_size=PADDING_SIZE, image_dims=(224,224), netname=classifier.name)
        else:
            print('Unknown test type')
            return 

        pda = PredDiffAnalyser(x_test[np.newaxis], classifier, sampler, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        pred_diff = pda.get_rel_vect(win_size=WIN_SIZE, overlap=OVERLAPPING)

        # for now
        sensMap = np.zeros(x_test_im.shape)

        print('diff shape', pred_diff[0].shape)

        print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
        # plot and save the results
        uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, top_catid, save_path)
        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, 385, save_path)
        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, 386, save_path)
        np.savez(save_path, *pred_diff)
        print('result:', save_path)
    



def main():

    # model = infoMobileNetV2(show=False)
    # resnet versions: 18,34,50,101,152
    # model = infoResNet(mVersion='resnet50')
    # model = infoAlexNet()
    # model = infoVgg16()
    model = infoCostumRSICD('./model/ood/react/resnet50_rsicd.pth', show=False)
    #experiment(model)
    viewImage(model)


if __name__=='__main__':
    main()

