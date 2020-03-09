import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.array(Y)
    P = np.array(P) # predictions
    return -np.sum(Y * np.log(P) + (1-Y) * np.log(1-P))

if __name__=="__main__":
    Y = [1,0,1,1] # ground truth
    P = [0.4, 0.6, 0.1, 0.5] # predictions
    output = cross_entropy(Y, P)
    print(output)