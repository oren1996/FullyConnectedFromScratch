from  hw1_204864532_q2_train import NN
import torchvision.datasets as dsets
import numpy as np
import pickle

def evaluate_hw1():    
    loss = pickle.load(open('model_q2.pkl','rb'))['loss']
    print(f'mean_loss_value_test: {loss}')
    
evaluate_hw1()