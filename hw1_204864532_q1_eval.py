from  hw1_204864532_q1_train import NN
import torchvision.datasets as dsets
import numpy as np
import pickle

def evaluate_hw1():
    test_dataset = dsets.MNIST(root='./data/',
                              train=False, 
                              download=True)
    
    nn = pickle.load(open('model_q1.pkl','rb'))
    accuracy = nn.compute_accuracy(test_data=test_dataset, numberOfLabels=10)
    print(f'accuracy: {accuracy}')
    

evaluate_hw1()