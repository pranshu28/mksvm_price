from sklearn.svm import SVR
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import itertools


def rbf(gamma=1.0):
	def rbf_fun(x1,x2):
		# print(gamma)
		return math.exp((np.linalg.norm(x1-x2))*(-1.0*gamma))
	return rbf_fun

def lin(offset=0):
	def lin_fun(x1,x2):
		return x1.dot(x2.transpose())+offset
	return lin_fun

def poly(power=2,offset=0):
	def poly_fun(x1,x2):
		# print(power,offset)
		return pow(x1.dot(x2.transpose())+offset,power)
	return poly_fun

def sig(alpha=1.0,offset=0):
	def sig_fun(x1,x2):
		return math.tanh(alpha*1.0*x1.dot(x2.transpose())+offset)
	return sig_fun

def kernel_matrix(x,kernel):
	# print(x.shape[0])
	mat=np.zeros((x.shape[0],x.shape[0]))
	for a in range(x.shape[0]):
		for b in range(x.shape[0]):
			mat[a][b]=kernel(x[a],x[b])
	return mat

def f_dot(kernel_mat1,kernel_mat2):
	# print(kernel_mat1,kernel_mat2)
	return (kernel_mat1.dot(kernel_mat2.transpose())).trace()

def A(kernel_mat1,kernel_mat2):
	return (f_dot(kernel_mat1,kernel_mat2))/(math.sqrt(f_dot(kernel_mat1,kernel_mat1)*f_dot(kernel_mat2,kernel_mat2)))

def beta_finder(x,y,kernel_list):
	y=np.matrix(y)
	yyT=(y.transpose()).dot(y)
	# print(yyT.shape)
	deno=sum([A(kernel_matrix(x,kernel),yyT) for kernel in kernel_list])
	betas=[A(kernel_matrix(x,kernel),yyT)/deno for kernel in kernel_list]
	return betas

def multi_kernel_maker(x,y,kernel_list):
	betas=[float(b) for b in beta_finder(x,y,kernel_list)]
	#print "	",betas
	def multi_kernal(x1,x2):
		mat=np.zeros((x1.shape[0],x2.shape[0]))
		for a in range(x1.shape[0]):
			for b in range(x2.shape[0]):
				mat[a][b]=sum([betas[i]*kernel(x1[a],x2[b]) for i,kernel in enumerate(kernel_list)])
		return mat
	return multi_kernal

#Input all the data
y_comm = pd.read_csv("Cleaned_prices.csv",header=None).as_matrix()
x_gdp = pd.read_csv("Cleaned.csv",header=None).as_matrix()
comms=y_comm.shape[1]
costs = {}
min_costs=[]
x_train = x_gdp[:-3,:]
x_test = x_gdp[-3:,:]
for comm in range(comms):
	y_train = y_comm[:-3,comm]
	y_act = y_comm[-3:,comm]
	kernels = [lin(),lin(2),poly(),poly(3),poly(4),rbf(),rbf(1.5),sig(),sig(1.5),rbf(2.0),sig()]
	multi_kernels = [mult for mult in itertools.combinations(kernels, 3)]

	# x=np.matrix([[1,2],[2,4],[3,6],[4,8],[5,10]])
	# y=np.matrix([[2],[4],[6],[8],[10]])

	#Run for each pair of kernel
	costs[comm]=[]
	err=99999
	err_ind=0
	for i,k_list in enumerate(multi_kernels):	
		kernel=multi_kernel_maker(x_train,y_train,k_list)
		svr=SVR(C=1.5,kernel=kernel)
		c=svr.fit(x_train,np.squeeze(y_train))
		y_test=svr.predict(x_test)
		error = abs(y_act-y_test)
		mean_error = np.mean(error)
		# print("Commodity Number ",comm,mean_error,k_list)
		if err>mean_error:
			err=mean_error
			err_ind=i
		costs[comm].append([mean_error,k_list])
	min_cost=costs[comm][err_ind][:]
	min_cost.insert(0,comm)
	print(min_cost)
	min_costs.append(min_cost)
	# print("Min error model of commodity ",comm,min_cost[0],min_cost[1])
pd.DataFrame(costs).to_csv('Costs.csv', sep=',')
pd.DataFrame(min_costs).to_csv('Comm_min_cost.csv', sep=',')
