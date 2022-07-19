#from sre_parse import CATEGORIES
# from statistics import mode
import numpy as np
import torch 
import torchvision as tv
from PIL import Image
import time
import os
import argparse


# pda imports 
import dataloader as udl 
import classifiers as ucls
import visualizer as uv
import sampler as sampling
from pred_diff_analyser import PredDiffAnalyser


from config import *
import matplotlib.pyplot as plt


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

    model = rn.ResNet(rn.Bottleneck, [3,4,6,3],  23)
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

    return {'model' : model, 'name' : "rsicd_jakob",'preprocess' : preprocess, 'display' : display, 'path' : path}


# def infoCostumRSICD(path, show=False):
#     '''
#     RSIDCD model info RESNET50
#     '''

#     model = rn.ResNet(rn.Bottleneck, [3,4,6,3],  23)
#     model.load_state_dict(torch.load(path))
#     if show: 
#         print('eval model: \n', model)

#     # define preprocess and disply corpping functions
#     preprocess = tv.transforms.Compose([
#         tv.transforms.Resize(256),
#         tv.transforms.CenterCrop(224),
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
#     display = tv.transforms.Compose([
#         tv.transforms.Resize(256),
#         tv.transforms.CenterCrop(224),
#         tv.transforms.ToTensor()])

#     return {'model' : model, 'name' : "rsicd_conrad",'preprocess' : preprocess, 'display' : display, 'path' : path}


def test_prob(classifier, x):
    '''
    test predict function 
    '''
    top_prob, top_catid = classifier.getTopCat(x[0:1])
    
    cat_prob = classifier.predict(x) 

    print(np.max(cat_prob), np.min(cat_prob))

def viewImage(model):

    dataLoader = udl.DataLoader(model=model, path=IMAGE_PATH)
    #dataLoader = udl.DataLoader(model=model)# , path='./img/')

    test_size = TESTS
    show      = True
    X_test,X_test_im, X_filenames = dataLoader.get_data(s_idx=IMG_IDX, b_size=test_size, set_size=test_size)

    classifier = ucls.Classifier(model, softmax=SOFTMAX)


    # classifier.softmax = False   
    # test_prob(classifier, X_test)
    # classifier.softmax = True
    # test_prob(classifier, X_test)

    for test_idx in range(test_size):

        x_test_img = X_test_im[test_idx]
        x_test     = X_test[test_idx:test_idx+1]

        print(x_test.shape)

        # res = classifier.predict(x_test)
        # print('RES SHAPE: ', res.shape)

        top_prob, top_catid = classifier.getTopCat(X_test[test_idx:test_idx+1])
        print('topP ', top_prob, 'ID', top_catid)
        # y_pred_label = classifier.categories[top_catid]
        # print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))
        if show:
            plt.imshow(x_test_img)
            plt.show()


def experiment(model):

    dataLoader = udl.DataLoader(model=model, path=IMAGE_PATH)

    test_size = TESTS
    show      = SHOW
    softmax = SOFTMAX
    X_test,X_test_im, X_filenames = dataLoader.get_data(s_idx=IMG_IDX, b_size=test_size)

    path_results = RESULT_PATH
    if not os.path.exists(path_results):
        os.makedirs(path_results)  

    classifier = ucls.Classifier(model, softmax=softmax, categroies=CATEGORIE_PATH)
    # Test: get prob of class from 
    #test_prob(classifier,X_test)


    for test_idx in range(test_size):

        x_test = X_test[test_idx]
        x_test_im = X_test_im[test_idx]
        x_test_path = X_filenames[test_idx]

        top_prob, top_catid = classifier.getTopCat(x_test[np.newaxis])

        y_pred_label = classifier.categories[top_catid]
        print('TEST: ', x_test_path)
        print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))


        # show image
        if show:
            plt.imshow(x_test_im)
            plt.show()

        start_time = time.time()

        if SAMPLE_STYLE == 'conditional':
            # save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}.jpg'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE,NUM_SAMPLES,PADDING_SIZE,classifier.name)
            save_path = path_results + '{}_{}_winSize{}_stride{}_sm{}_{}.jpg'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE, STRIDE,softmax,classifier.name)
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
        # print('pred_diff',  len(pred_diff))
        # for i in range(23):
        #     uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, i, save_path)

        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, 385, save_path)
        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, 386, save_path)
        
        uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, top_catid, save_path)
        np.savez(save_path, *pred_diff)
        print('result:', save_path)
    



def visualize(model, im_idx=IMG_IDX, test_classes=5):
    '''
    model: model for classifcation to get top cat id
    im_idx : get specific image form folder
    test_classes: number of random classes to be shown
    '''

    test_size = TESTS
    show      = SHOW
    softmax = SOFTMAX

    path_results = RESULT_PATH


    dataLoader = udl.DataLoader(model=model, path=IMAGE_PATH)

    X_test,X_test_im, X_filenames = dataLoader.get_data(s_idx=im_idx, b_size=test_size)

    classifier = ucls.Classifier(model, softmax=softmax, categroies=CATEGORIE_PATH)


    for test_idx in range(test_size):

        x_test = X_test[test_idx]
        x_test_im = X_test_im[test_idx]
        x_test_path = X_filenames[test_idx]

        top_probs, top_catids = classifier.getTopCats(x_test[np.newaxis],top=5)
        top_prob   = top_probs[0][0].item()
        top_catid  = top_catids[0][0].item()
        y_pred_label = classifier.categories[top_catid]



        load_path = path_results + '{}_{}_winSize{}_stride{}_sm{}_{}.jpg.npz'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE, STRIDE,softmax,classifier.name)

        print('Try to load: ', load_path)


        npFile = np.load(load_path)
        # print('Arrays in file:', npFile.files)
        pred_diff = npFile['arr_0']
        # print('array: ', pred_diff.shape)
        # not use, simply set to zero for now
        sensMap = np.zeros(x_test_im.shape)

        # get min max values from show class
        p_class = pred_diff[:,top_catid]
        print('MIN: {}/MAX: {}'.format(np.min(p_class), np.max(p_class)))

        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, top_catid)
        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, 385)
        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, 386)

        for cid in range(test_classes):
            catid = top_catids[0][cid].item()
            uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, catid)


        # for catid in np.random.randint(1000, size=test_classes):
        #     uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, catid)

        

def main():

    parser = argparse.ArgumentParser(description='Prediction Difference Analysis')
    parser.add_argument('-v', '--visualize',action="store_true", help='generate testbench files')
    
    args = parser.parse_args()


    if MODEL == 'mnv2':
        model = infoMobileNetV2(show=False)
    elif MODEL == 'resnet50':
        # resnet versions: 18,34,50,101,152
        model = infoResNet(mVersion='resnet50')
    elif MODEL == 'alexnet':
        model = infoAlexNet()
    elif MODEL == 'vgg16':
        model = infoVgg16()
    elif MODEL == 'c_jakob':
        model = infoCostumRSICD('./model/ood/react/resnet50_rsicd.pth', show=False)
    
    if args.visualize:
        visualize(model)
        return 
    else: 
        experiment(model)
        return 

    # viewImage(model)
    


if __name__=='__main__':
    main()

