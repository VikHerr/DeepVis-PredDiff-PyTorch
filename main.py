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



def getMobileNetV2(show=False):
    '''
    get mobelNetV2 with pretrained weights on image net in eval mode
    '''
    model = tv.models.mobilenet_v2(pretrained=True)
    if show: 
        print('eval model: \n', model)

    model.eval()

    return model

def test_prob(classifier, x):
    '''
    test predict function 
    '''
    top_prob, top_catid = classifier.getTopCat(x[0:1])
    
    cat_prob = classifier.predict(x) 

    print(cat_prob.shape)


def experiment(model):

    dataLoader = udl.DataLoader()

    test_size = TESTS
    show      = False
    X_test,X_test_im, X_filenames = dataLoader.get_imagenet_data(b_size=test_size)

    path_results = './results/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)  

    classifier = ucls.Classifier()
    # Test: get prob of class from 
    test_prob(classifier,X_test)


    for test_idx in range(test_size):

        top_prob, top_catid = classifier.getTopCat(X_test[test_idx:test_idx+1])
        y_pred_label = classifier.categories[top_catid]
        print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))

        # show image
        if show:
            plt.imshow(X_test_im[test_idx])
            plt.show()

        start_time = time.time()

        if SAMPLE_STYLE == 'conditional':
            save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}.jpg'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE,NUM_SAMPLES,PADDING_SIZE,classifier.name)
            sampler = sampling.ConditionalSampler(win_size=WIN_SIZE, padding_size=PADDING_SIZE, image_dims=(224,224), netname=classifier.name)
        else:
            print('Unknown test type')
            return 

        pda = PredDiffAnalyser(X_test[0:1], classifier, sampler, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        pred_diff = pda.get_rel_vect(win_size=WIN_SIZE, overlap=OVERLAPPING)

        # for now
        sensMap = np.zeros(X_test_im[test_idx].shape)

        # plot and save the results
        uv.plot_results(X_test[test_idx], X_test_im[test_idx], sensMap, pred_diff[0], classifier.categories, top_catid, save_path)
        np.savez(save_path, *pred_diff)
        print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
    


def main():

    model = getMobileNetV2()
    
    batch_size = 128 
    experiment(model)



if __name__=='__main__':
    main()

