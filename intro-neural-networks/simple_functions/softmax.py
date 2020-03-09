"""
For a given list, calculate the Softmax
"""
import numpy as np

def softmax(in_array):
    return np.exp(in_array)/np.sum(np.exp(in_array))
if __name__=="__main__":
    L = [5,6,7]
    L_array = np.array(L)
    L_array_softmax = softmax(L_array)
    print('input:', L_array)
    print('softmax:', L_array_softmax)
    print('sum: ', np.sum(L_array_softmax))
