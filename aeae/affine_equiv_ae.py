import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from utils import *

def change_embed(sigma):
    '''
    function 't' defined in the paper Affine Equivariant Autoencoder https://www.ijcai.org/Proceedings/2019/0335.pdf
    Input:
        sigma = affine transformation parameters (length =5)
    Output:
        t(sigma) = vector to be added to the embedding of original image
    '''
    return torch.Tensor([0]*10 + [sigma[0]/100,sigma[1]/10,sigma[2]/10,sigma[3]-1,sigma[4]/100]).cuda()

def param2affine(sigma):
    '''
    Conversion of affine parameter into matrix form through the use of module 'affine'
    sigma[0] = degree of anti-clockwise rotation (unit = degrees)
    sigma[1] = translation rightwards
    sigma[2] = translation down
    sigma[3] = scaling factor
    sigma[4] = degree of shear transformation
    
    Input:
        sigma = affine transformation parameter
    Output:
        (2,3) np.array representing transformation
    '''
    aff_mat = Affine.translation(sigma[1],sigma[2]) * Affine.scale(sigma[3]) * Affine.rotation(sigma[0])* Affine.shear(sigma[4],sigma[4])
    return np.array(aff_mat)[:6].reshape(2,3)

class AEAE(nn.Module):
    def __init__(self, dim):
        super(AEAE,self).__init__()
        self.fc1 = nn.Linear(dim,500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,2000)
        self.fc4 = nn.Linear(2000,15)
        self.fc5 = nn.Linear(15,2000)
        self.fc6 = nn.Linear(2000,500)
        self.fc7 = nn.Linear(500,500)
        self.fc8 = nn.Linear(500,dim)
        
    def forward(self,x,sigma):
        '''
        Forward propagation of the model. This consists of three different paths
            1. original image encode-decode
            2. transformed image encode-decode
            3. original image's embedding + t_sigma(residual) decode
        Input:
            x = original image
            sigma = affine transformation parameter 
        Output:
            trans_x = transformed image (for loss calculation)
            x_pred =  reconstructed original image
            trans_x_pred = reconstructed transformed image
            trans_x_pred_2 = reconstructed tranformed image 2
            original_embedding = embedding of original image
            trans_embedding = embedding of transformed image
            trans_embedding_2 = embedding of original image + residual
        '''
        #original shape
        original_dim = x.shape
        x = x.squeeze(1)
        #transform image
        affine_mat = param2affine(sigma)
        tmp = [cv2.warpAffine(np.float32(x[i].cpu()),affine_mat,(x.size(1),x.size(2))) for i in range(x.size(0))]
        trans_x = torch.Tensor(tmp).cuda()
        # original image encode-decode
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # save embedding
        original_embedding = x.clone()
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.sigmoid(self.fc8(x))
        x_pred = x.view(original_dim)

        # transformed image encode-decode
        trans_x_vec = trans_x.view(trans_x.size(0), -1)
        trans_x_vec = F.relu(self.fc1(trans_x_vec))
        trans_x_vec = F.relu(self.fc2(trans_x_vec))
        trans_x_vec = F.relu(self.fc3(trans_x_vec))
        trans_x_vec = self.fc4(trans_x_vec)
        trans_embedding = trans_x_vec.clone()
        trans_x_vec = F.relu(self.fc5(trans_x_vec))
        trans_x_vec = F.relu(self.fc6(trans_x_vec))
        trans_x_vec = F.relu(self.fc7(trans_x_vec))
        trans_x_vec = F.sigmoid(self.fc8(trans_x_vec))
        trans_x_pred = trans_x_vec.view(original_dim)

        #tranformed embedding decode
        t_sig = change_embed(sigma)
        trans_embedding_2 = original_embedding + t_sig
        trans_pred = F.relu(self.fc5(trans_embedding_2))
        trans_pred = F.relu(self.fc6(trans_pred))
        trans_pred = F.relu(self.fc7(trans_pred))
        trans_pred = F.sigmoid(self.fc8(trans_pred))
        trans_x_pred_2 = x.view(original_dim)

        return trans_x, x_pred, trans_x_pred,trans_x_pred_2 , original_embedding, trans_embedding, trans_embedding_2

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(AEAE, self).parameters())

