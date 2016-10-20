from sklearn.svm import SVR
import numpy as np
import math
def rbf(x1,x2,gamma=1.0):
	return (np.linalg.norm(x1-x2))*(-1.0*gamma)
def lin(x1,x2,offset=0):
	return x1.dot(x2)+offset
def poly(x1,x2,power=3,offset=0):
	return pow(x1.dot(x2)+offset,power)
def sig(x1,x2,alpha=1.0,offset=0):
	return math.tanh(alpha*1.0*x1.dot(x2)+offset)
