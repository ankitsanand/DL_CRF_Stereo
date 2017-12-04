import opengm
import torch
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CRF(autograd.Function):
  """
    Takes the unary and pairwise potentials, do CRF inference over the given graphical model and return the labels
    Args:
      
      unary: unary potentials (b x k x r x c)
      pairwise: Format To be decided
    Return:
      Labels - (b x r x c)
  """
  def __init__(self,true_labels):
    super(CRF, self).__init__()
    self.labels=true_labels
    



  def forward(self, unary_pots):
    """ Receive input tensor, return output tensor"""
    self.save_for_backward(unary_pots)
    
    print("In forward")
    b,r,c,k = unary_pots.size()
    
    if(False):
        if torch.cuda.is_available():
            unaries = unary_pots.cpu().numpy()
        else:
            unaries = unary_pots.numpy()

        unaries = unaries.reshape([b*r*c,k])
        numVar = r*c
        
        gm=opengm.gm(np.ones(numVar,dtype=opengm.label_type)*k)
        uf_id = gm.addFunctions(unaries)
        potts = opengm.PottsFunction([k,k],0.0,0.4)
        pf_id = gm.addFunction(potts)

        vis=np.arange(0,numVar,dtype=np.uint64)
        # add all unary factors at once
        gm.addFactors(uf_id,vis)
        # add pairwise factors 
        ### Row Factors
       
        for i in range(0,r):
            for j in range(0,c-1):
                gm.addFactor(pf_id,[i*c+j,i*c+j+1])
        ### Column Factors
        for i in range(0,r-1):
            for j in range(c):
                gm.addFactor(pf_id,[i*c+j,(i+1)*c+j])
        print("Graphical Model Constructed")
        inf=opengm.inference.AlphaExpansionFusion(gm)
        inf.infer()
        labels=inf.arg()
        
        return torch.from_numpy(np.asarray(labels).astype('float'))
    else:
        return torch.zeros(b,r,c)
    #return torch.from_numpy(numpy.random.rand(r,c,numLabels))
    

  def backward(self,grad_output):
    """Calculate the gradients of left and right"""
    print("Entering Backward Pass Through CRF\n Max Grad Outputs",torch.max(grad_output),torch.min(grad_output))
    unary_pots, = self.saved_tensors
    true_labels = self.labels
    true_labels = true_labels.data.cpu().numpy()
    #unary_pots = unary_pots_temp[:,:,0:10,0:10]
    b,r,c,k = unary_pots.size()
    
#     # r=10
#     # c=10
    # print(unary_pots.size())
    # print(true_labels.shape)
    gamma = 0.1
    tau = 10
    unary_flat = unary_pots.contiguous().view([b*r*c,k])
    numVar = r*c
    index_arr=torch.zeros(r*c,k)
    for i in range(k):
        index_arr[:,i] = i
    for j in range(numVar):      
        for i in range(k):
            unary_flat[j,i] = unary_flat[j,i] - gamma* min(abs(i-true_labels[j]),tau)
      
    if torch.cuda.is_available():
        unaries = unary_flat.cpu().numpy()
    else:
        unaries = unary_flat.numpy()
    gm=opengm.gm(np.ones(numVar,dtype=opengm.label_type)*k)
    uf_id = gm.addFunctions(unaries)
    potts = opengm.PottsFunction([k,k],0.2,1.0)
    pf_id = gm.addFunction(potts)

    vis=np.arange(0,numVar,dtype=np.uint64)
        # add all unary factors at once
    gm.addFactors(uf_id,vis)
        # add pairwise factors 
        ### Row Factors
   
    for i in range(0,r):
        for j in range(0,c-1):
            gm.addFactor(pf_id,[i*c+j,i*c+j+1])
        ### Column Factors
    for i in range(0,r-1):
        for j in range(c):
            gm.addFactor(pf_id,[i*c+j,(i+1)*c+j])
    print("Graphical Model Constructed")
    infParam = opengm.InfParam(steps=5)
    inf=opengm.inference.AlphaExpansionFusion(gm,parameter=infParam)
    inf.infer()

    print("Inference done")
    del_x_bar = inf.arg()
    sub_grad_unaries = np.zeros((b*r*c,k))
    energy = 0
    for i in range(numVar):
        energy = energy - unaries[i][del_x_bar[i]] + unaries[i][int(true_labels[i])] + gamma* min(abs(del_x_bar[i]-true_labels[i]),tau)
    for i in range(0,r):
        for j in range(0,c-1):
            if(del_x_bar[i*c+j]==del_x_bar[i*c+j+1]):
                energy = energy - 0.2
            else:
                energy = energy - 1.0
            if(true_labels[i*c+j]==true_labels[i*c+j+1]):
                energy = energy +0.2
            else:
                energy = energy + 1.0
    for i in range(0,r-1):
        for j in range(c):
            if(del_x_bar[i*c+j]==del_x_bar[(i+1)*c+j]):
                energy = energy - 0.2 
            else:
                energy = energy - 1.0
            if(true_labels[i*c+j]==true_labels[i*c+j+1]):
                energy = energy + 0.2
            else:
                energy = energy + 1.0

    print("Energy",energy)
    for i in range(numVar):  
        sub_grad_unaries[i,int(del_x_bar[i])] = -1
        #print true_labels[i]
        if(true_labels[i]==-1):
            continue
        sub_grad_unaries[i,int(true_labels[i])] = 1 

    grad_in = sub_grad_unaries.reshape([b,r,c,k]) 
   
   
    print("Leaving Backward through CRF\n Min Grad Inputs",torch.max(grad_output),torch.min(grad_output))
    
    return torch.from_numpy(grad_in).type('torch.cuda.FloatTensor')
# width=100
# height=200
# numVar=width*height
# numLabels=2
# # construct gm
# gm=opengm.gm(np.ones(numVar,dtype=opengm.label_type)*numLabels)
# # construct an array with all numeries (random in this example)
# unaries=np.random.rand(width,height,numLabels)
# # reshape unaries is such way, that the first axis is for the different functions
# unaries2d=unaries.reshape([numVar,numLabels])
# # add all unary functions at once (#numVar unaries)
# fids=gm.addFunctions(unaries2d)
# # numpy array with the variable indices for all factors
# vis=np.arange(0,numVar,dtype=numpy.uint64)
# # add all unary factors at once
# gm.addFactors(fids,vis)
# print("Graphical Model Constructed")