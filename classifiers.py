# Classifier functions

import torch 
import torchvision as tv

class Classifier:

    def __init__(self, model, path="model/ilsvrc_2012_labels.txt", softmax=True, categroies="model/imagenet_classes.txt"):
        '''
        path: path to result categories
        '''

        self.name = model['name']
        # self.cPath = model['path'] # only for costum models
        self.gpu = torch.cuda.is_available()
        self.model = self.__getClassifier(model)
        self.categories = self.__getCategories(categroies)
        self.softmax    = softmax

    def predict(self,x):

        probabilities = self.forward_pass(x)
        
        probabilities = probabilities.cpu().detach().numpy()

        #cat_prob = self.getCatProb(probabilities,category_idx)

        return probabilities



    def forward_pass(self, x):
        '''
        forward pass of pytorch net
        '''
        if self.gpu:
            x = x.to('cuda')

        with torch.no_grad():
            output = self.model(x)

        if self.softmax:
            probabilities = torch.nn.functional.softmax(output, dim=1)
        else:
            probabilities = output

        return probabilities

    def getCatProb(self,prob,cat_idx):
        '''
        get probability of specific categorie
        '''

        return prob[:,cat_idx]
          
 
    def getTopCats(self,x, top=1):
        '''
        get top categorys
        '''
        temp = self.softmax
        # ensure softmax here
        self.softmax = True
        if self.gpu:
            x = x.to('cuda')

        prob = self.forward_pass(x)
        self.softmax = temp

        return torch.topk(prob, top)
    
    def getTopCat(self,x):
        '''
        get idx of top category
        '''
        top_probs, top_catids = self.getTopCats(x,top=1)
        return top_probs[0][0].item(), top_catids[0][0].item()
        
    def report(self):
        print('eval model: \n', self.model)

    def __getClassifier(self, modelInfo):
        '''
        define classifier model
        '''
        # classifier is selected from toplevel now
        # if(self.name=='moblienet_v2'):
        #     model = tv.models.mobilenet_v2(pretrained=True)
        #     # switch to inferance mode (This should be done for all models)
        # elif('resnet' in self.name):

        #     if self.name == 'resnet50':
        #         model = tv.models.resnet50(pretrained=True)
        #     elif self.name == 'resnet18':
        #         print('load resnet18')
        #         model = tv.models.resnet18(pretrained=True)
        #     else:
        #         assert False, 'invaled resnet version'

        #     model = tv.models.resnet50(pretrained=True)
        # elif(self.name == 'alexnet'):
        #     model = tv.models.alexnet(pretrained=True)

        # elif(self.name == 'vgg16'):
        #     model = tv.models.vgg16(pretrained=True)
        # elif(self.name == 'costum'):
        #     model = torch.load(self.cPath)
        # else:
        #     assert False, 'unkown model name!'

        # take model from toplevel
        model = modelInfo['model']
        model.eval()

        if self.gpu:
            model.to('cuda')

        return model


    def __getCategories(self, path):
        '''
        load categories for classifier
        '''
        with open(path, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        return categories



        

        