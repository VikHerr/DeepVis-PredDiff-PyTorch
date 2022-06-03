# -*- coding: utf-8 -*-
"""
@author: Viktor Herrmann
most of the code is adaped from Luisa Zintgraf
"""

import numpy as np
import time
import torch

from config import STRIDE

class PredDiffAnalyser:
    '''
    This class implements the prediction difference analysis, i.e., a method
    which estimates how important the individual input features are to a 
    (already trained) classifier, given a specific input to the classifier.    
    To this end, a relevance is estimated which is of the same size as the 
    input and reflects the importance of each feature.
    
    Note: this version implements the method for RGB-image classification!
    However, the method can be used with any kind of data.
    Also note that we assume that the color channels are along axis 0, as it 
    is common with convolutional neural networks.
    '''
        
    def __init__(self, x, classifier, sampler, num_samples=10, batch_size=10, prob_tar=True):
        '''
        Input:  
            x           the feature vector for which we want to make the analysis (can be a hidden layer!)
                        Has to be numpy array of the dimension that fits to targetFunc
            tar_func    the target function, can be the output of classifier or intermediate layer
                        (must take x as input, keep this in mind when starting at intermediate layers!)
            num_samples the number of samples used for marginalising out features
            batch_size  batch size for caffe network (in tar_func)
            prob_tar    boolean, indicates if the target values are probabilities 
                        (not necessarily the case when we look at hidden nodes)
        '''
        

        # classifier resuts
        self.tar_func = classifier.predict
        self.classifier = classifier
        self.true_tar_val = self.tar_func(x)  # true network state for the given input 
        #self.true_tar_val = self.tar_func(torch.tensor(x)) 
        #print('target: ', classifier.categories[self.true_tar_idx],'| prob %.4f ' %(self.true_tar_val))

        # inputs
        self.x = np.copy(x.cpu().detach().numpy())
        self.sampler = sampler
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.prob_tar = prob_tar
        
        # some other useful values
        self.num_feats = len(self.x.ravel())//3  # we make the analysis not per color channel, 
                                                # but for all channels at once,
                                                # therefore we divide the number of features by 3
        
        self.num_blobs   = 1# len(self.true_tar_val) 
        self.num_metrics = 2                    # the number of metrics we use for evaluating
                                                # the prediction difference (avg and max of 
                                                # the weight of evidence per feature map)


        self.tests_per_batch = int(self.batch_size//self.num_samples) # rounds down      
        
        # drop the first dimension of the elements in the true target value list,
        # since it is not necessary (since we only forwarded a single feature vector)
        #self.true_tar_val = [t[0] for t in self.true_tar_val]

   
 
#%%
# -------------------- METHOD RETURNING EXPLANATIONS --------------------------  
        
    def get_rel_vect(self, win_size, overlap=True):
        """
        Main method to use, will return a relevance vector.
        Input:  win_size    the window size (k in alg. 1)
                overlap     whether the windows should be overlapping, default is True
        Output: rel_vects   the relevance vectors, dimensions are:
                            - number of features (input)
                            - number of outputs (usually output layer, can be hidden layer)
                            to interpret the result, look at one output (e.g., the predicted class)
                            and visualise the input features in some way
        """
        
        # create array for relevance vectors, each element has dimensions (num_feats)*blobdimension
        # where the relevance of each feature on the different activations in that blob is stored
        rel_vects = [np.zeros((self.num_feats, self.true_tar_val[0].shape[-1]), dtype=np.float64)]
        #rel_vects = [np.zeros((self.num_feats, self.true_tar_val[b].shape[0]), dtype=np.float64) for b in range(self.num_blobs)]


        # a counts vector to keep track of how often a feature is marginalised out
        counts = np.zeros((self.num_feats), dtype=np.int)

        # a matrix where each entry reflects the index in the flattened input (image)
        all_feats = np.reshape([i for i in range(self.num_feats*3)], self.x.shape)
        
        if overlap == 'full':
            windows = np.zeros((self.tests_per_batch, win_size*win_size*3), dtype=int)
            win_idx = 0
            for i in range(self.x.shape[-1]-win_size+1): # rows
                start_time = time.time()
                for j in range(self.x.shape[-2]-win_size+1): # columns
                    # get the window which we want to simulate as unknown
                    window = all_feats[0,:,i:i+win_size,j:j+win_size].ravel()                    
                    windows[win_idx] = window
                    win_idx += 1
                    if win_idx==self.tests_per_batch:
                        # evaluate the prediction difference
                        pred_diffs = self._get_rel_vect_subset(windows)
                        for w in range(self.tests_per_batch):                            
                            window = windows[w]
                            for b in range(self.num_blobs):
                                rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                            counts[window[window<self.num_feats]] += 1
                        win_idx = 0
                print ("row {}/{} took: --- {:.4f} seconds --- ".format(i,self.x.shape[1]-win_size+1,(time.time() - start_time)))
#                sys.stdout.write("\033[F")
                
            # evaluate the rest that didn't fill last batch
            pred_diffs = self._get_rel_vect_subset(windows[:win_idx+1])
            for w in range(win_idx+1):
                window = windows[w]
                for b in range(self.num_blobs):
                    rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                counts[window[window<self.num_feats]] += 1
        
        elif overlap == 'stride':
            windows = np.zeros((self.tests_per_batch, win_size*win_size*3), dtype=int)
            win_idx = 0
            stride = STRIDE
            for i in range(self.x.shape[-1]//stride - (stride+1)): # rows
                start_time = time.time()
                for j in range(self.x.shape[-2]//stride- (stride+1)): # columns
                    # get the window which we want to simulate as unknown
                    window = all_feats[0,:,i*stride:i*stride+win_size,j*stride:j*stride+win_size].ravel()
                    windows[win_idx] = window
                    win_idx += 1
                    if win_idx==self.tests_per_batch:
                        # evaluate the prediction difference
                        pred_diffs = self._get_rel_vect_subset(windows)
                        for w in range(self.tests_per_batch):
                            window = windows[w]
                            for b in range(self.num_blobs):
                                rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                            counts[window[window<self.num_feats]] += 1
                        win_idx = 0
                print ("row {}/{} took: --- {:.4f} seconds --- ".format(i,self.x.shape[1]//stride-1,(time.time() - start_time)))
            # evaluate the rest that didn't fill last batch

            pred_diffs = self._get_rel_vect_subset(windows[:win_idx+1])
            for w in range(win_idx+1):
                window = windows[w]
                for b in range(self.num_blobs):
                    rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                counts[window[window<self.num_feats]] += 1

                    
        else: 
            windows = np.zeros((self.tests_per_batch, win_size*win_size*3), dtype=int)
            win_idx = 0
            for i in range(self.x.shape[-1]//win_size): # rows
                start_time = time.time()
                for j in range(self.x.shape[-2]//win_size): # columns
                    # get the window which we want to simulate as unknown
                    window = all_feats[0,:,i*win_size:i*win_size+win_size,j*win_size:j*win_size+win_size].ravel()
                    windows[win_idx] = window
                    win_idx += 1
                    if win_idx==self.tests_per_batch:
                        # evaluate the prediction difference
                        pred_diffs = self._get_rel_vect_subset(windows)
                        for w in range(self.tests_per_batch):
                            window = windows[w]
                            for b in range(self.num_blobs):
                                rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                            counts[window[window<self.num_feats]] += 1
                        win_idx = 0
                print ("row {}/{} took: --- {:.4f} seconds --- ".format(i,self.x.shape[1]/win_size-1,(time.time() - start_time)))
#                sys.stdout.write("\033[F")
                
            # evaluate the rest that didn't fill last batch
            pred_diffs = self._get_rel_vect_subset(windows[:win_idx+1])
            for w in range(win_idx+1):
                window = windows[w]
                for b in range(self.num_blobs):
                    rel_vects[b][window[window<self.num_feats]] += pred_diffs[b][w]
                counts[window[window<self.num_feats]] += 1

                   
        # get average relevance of each feature
        for b in range(self.num_blobs):
            rel_vects[b][counts!=0] = ( rel_vects[b][counts!=0]/counts[counts!=0][:,np.newaxis] ).astype(np.float16)

            
        return rel_vects
       
        
#%%        
# -------------------------- SAMPLING METHODS ---------------------------------
        
        
    def _get_rel_vect_subset(self, feature_sets):
        """
        Returns the relevance vector, given the features that are unknown.
        Input:  notGivenFeats   indices of the raveled (!) feature vector
                                that are unknown
        """  

        for feature_set in feature_sets:
            # make variable notGivenFeats ready to use it as an index
            if type(feature_set)==set or type(feature_set)==list:
                feature_set = np.array(list(feature_set)) 
            if type(feature_set)==int:
                feature_set = np.array([feature_set]) 
            # if there are no features given, return a contribution of 0
            if (type(feature_set)==list and feature_set==[]) or (type(feature_set)!=int and len(feature_set)==0):
                print('feature set error')
                return 0
        
        # for each data point in X, replace the (not-given-)feature with the value for it seen in X
        x_new = np.zeros((self.tests_per_batch,self.num_samples,len(self.x.ravel())))
        x_new[:] = np.copy(self.x).ravel()[np.newaxis]

        for f in range(len(feature_sets)):
            temp_sample = self.sampler.get_samples(feature_sets[f], self.x, self.num_samples).T
            f_idx = feature_sets[f].ravel()
            #temp = np.copy(x_new[f, :, f_idx])
            x_new[f, :, f_idx] = temp_sample
            #diff =  temp - temp_sample
            #diiff = diff + 1


        # reshape x and set as pytorch tensor
        b_s,f_ch,f_x,f_y = self.x.shape
        x_tensor_shape = (self.tests_per_batch*self.num_samples, f_ch,f_x,f_y)
        x_tensor = torch.tensor(x_new.reshape(x_tensor_shape),dtype=torch.float)

        # get prediction for the altered x-values
        tarVals = self.tar_func(x_tensor)
        #tarVals = self.tar_func(x_new.reshape((self.tests_per_batch*self.num_samples,-1)))

        # for b in range(self.num_blobs):
        #     tarVals[b] = tarVals[b].reshape((self.tests_per_batch,self.num_samples,-1))
        tarVals = tarVals.reshape((self.tests_per_batch,self.num_samples,-1))

        # evaluate the prediction difference
        rel_vect = self._evaluate_prediction_difference(tarVals)

        return rel_vect
          
 
#%%       
# -------------- EVALUATION OF THE PREDICTION DIFFERENCE ----------------------        
                
    def _evaluate_prediction_difference(self, tarVals):
        '''
        Evaluating the prediction difference using the weight of evidence 
        Input:  tarVals     the values of all the blobs for several altered inputs
                            length is self.num_blobs, dimensions of each blob 
                            are (num_featsures)*(shape of blob)
        '''
        # average over all predictions received by using altered input values
        # we will save a value per feature map (instead of for each activation)
        # therefore we loog at the avg over all activations, and the max
        prediction_diffs = []
        # For the laplace correction, we need the number of training instances
        IMAGENET_TRAINSIZE = 100000
        pred_diffs = np.zeros((self.tests_per_batch,tarVals.shape[-1]))
        for t in range(self.tests_per_batch):
            avgP = np.average(tarVals[t], axis=0)
            # if we deal with probabilities, i.e., the last blobs, use this (always):
            # do a laplace correction to avoid problems with zero probabilities
            tarVal_laplace = (self.true_tar_val*IMAGENET_TRAINSIZE+1)/(IMAGENET_TRAINSIZE+1) #len(self.true_tar_val))
            avgP_laplace = (avgP*IMAGENET_TRAINSIZE+1)/(IMAGENET_TRAINSIZE+1)# len(self.true_tar_val))
            # calculate the odds for the true targets and  the targets with some features marginalised out
            oddsTarVal = np.log2(tarVal_laplace/(1-tarVal_laplace))
            oddsAvgP = np.log2(avgP_laplace/(1-avgP_laplace))
            # take average over feature maps
            pd = oddsTarVal-oddsAvgP
            # avg/max for the feature maps if we have feature maps in conv layers
            pd = pd.reshape((pd.shape[0], -1))
            pred_diffs[t] = np.average(pd, axis=1) # will only have an effect for convolutional layers
            # return the weight of evidence
        prediction_diffs.append(pred_diffs)
        return prediction_diffs
           
    # def _evaluate_prediction_difference(self, tarVals):
    #     '''
    #     Evaluating the prediction difference using the weight of evidence 
    #     Input:  tarVals     the values of all the blobs for several altered inputs
    #                         length is self.num_blobs, dimensions of each blob 
    #                         are (num_featsures)*(shape of blob)
    #     '''
    #     # average over all predictions received by using altered input values
    #     # we will save a value per feature map (instead of for each activation)
    #     # therefore we loog at the avg over all activations, and the max
    #     prediction_diffs = []
    #     # For the laplace correction, we need the number of training instances
    #     IMAGENET_TRAINSIZE = 100000
    #     for b in range(self.num_blobs): 
    #         pred_diffs = np.zeros((self.tests_per_batch,tarVals[b].shape[-1]))
    #         for t in range(self.tests_per_batch):
    #             avgP = np.average(tarVals[b][t], axis=0)
    #             # if we deal with probabilities, i.e., the last blobs, use this:
    #             if b==(self.num_blobs-1):
    #                 # do a laplace correction to avoid problems with zero probabilities
    #                 tarVal_laplace = (self.true_tar_val[b]*IMAGENET_TRAINSIZE+1)/(IMAGENET_TRAINSIZE+len(self.true_tar_val[b]))
    #                 avgP_laplace = (avgP*IMAGENET_TRAINSIZE+1)/(IMAGENET_TRAINSIZE+len(self.true_tar_val[b]))
    #                 # calculate the odds for the true targets and  the targets with some features marginalised out
    #                 oddsTarVal = np.log2(tarVal_laplace/(1-tarVal_laplace))
    #                 oddsAvgP = np.log2(avgP_laplace/(1-avgP_laplace))
    #                 # take average over feature maps
    #                 pd = oddsTarVal-oddsAvgP
    #             # if we do not deal with probabilities, we just return the distance to the average
    #             else:
    #                 pd = self.true_tar_val[b] - avgP    
    #             # avg/max for the feature maps if we have feature maps in conv layers
    #             pd = pd.reshape((pd.shape[0], -1))
    #             pred_diffs[t] = np.average(pd, axis=1) # will only have an effect for convolutional layers
    #             # return the weight of evidence
    #         prediction_diffs.append(pred_diffs)
    #     return prediction_diffs
  
  