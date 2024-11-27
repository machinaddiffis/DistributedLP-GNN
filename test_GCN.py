from model import *
import torch 
import gzip
import pickle
import os
import random
import torch
import torch.optim as optim
from model import GCN
import matplotlib.pyplot as plt
import numpy as np
import argparse
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-size','--size',type=int,default=1500)#radius
parser.add_argument('-L','--L',type=int,default=5)#radius
args = parser.parse_args()
#inference
size=args.size
L=args.L




if torch.cuda.is_available():
	device = "cuda"
	
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"


def max_product(M,p):
	m,n=M.shape
    
	P=torch.tile(p.unsqueeze(1), (1, m)).to(device)
	PM=torch.mul(M.to(device),torch.t(P).to(device))
	q=torch.max(PM,dim=1).values 

	return q


#dateset seting:



print('==> Building model..')


net=GCN(2,2,64).to(device)
model_name=f"./pretrain/best_GCN_size{size}_num100_L5.pth"
net.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
net.eval()

#optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer.zero_grad()

def test(size,task):
	R_primal=[]
	R_dual=[]
	train_loss=0
	if task==1:# test set
		training_data_len=100
		pointer=100
	else: # train set
		training_data_len=100
		pointer=0

	for iter in range(training_data_len):
		iter=iter+pointer
		with open(f'./instance/size_{size}/LPinstance_{size}_{iter}.pkl', 'rb') as f:
			data_list = pickle.load(f)
		# print(f"{iter}------------------------")
		A=torch.tensor(data_list[0],dtype=torch.float32).to(device)
		M, N = A.shape


		x = torch.zeros(size=(N, 2))
		y = torch.zeros(size=(M, 2))

		x = torch.as_tensor(x, dtype=torch.float32).to(device)
		y = torch.as_tensor(y, dtype=torch.float32).to(device)

		pred_primal, pred_dual = net(A, x, y)

		primal=torch.tensor(data_list[1],dtype=torch.float32).to(device)
		dual=torch.tensor(data_list[2],dtype=torch.float32).to(device)

		x_dot=resortation_1(A,pred_primal)
		y_dot=resortation_2(A,pred_dual)

	
		if(y_dot==None):

			continue
		if (x_dot == None):

			continue
		rp=abs(torch.sum(x_dot)-torch.sum(primal))/torch.sum(primal)

		rd=abs(torch.sum(y_dot)-torch.sum(dual))/torch.sum(dual)
	
		
		R_primal.append(rp)
		R_dual.append(rd)
		
	print(torch.mean(torch.tensor(R_primal)))
	# print("------------------------")
	print(torch.mean(torch.tensor(R_dual)))
	R_primal=torch.mean(torch.tensor(R_primal))
	R_dual=torch.mean(torch.tensor(R_dual))
	return R_primal,R_dual

def resortation_1(A,x):
	M,N=A.shape
	x=x.squeeze(1)
	ones=torch.ones(N).to(device)
	zeros=torch.zeros(M).to(device)
	x=torch.max(zeros,torch.min(ones,x))
	for i in range(M):
		term=torch.sum(A[i]*x)
		if term >=1:
			nx=torch.where(A[i]!=0)
			x[nx]=x[nx]/term

	return x

def resortation_2(A,y):
	M,N=A.shape
	ones=torch.ones(M).to(device)
	y=y.squeeze(1)
	eps=1e-5	
	y=torch.max(ones*eps,torch.min(ones,y))
	for j in range(N):
		term=torch.sum(torch.t(A)[j]*y)
		if term <=1:
			nx=torch.where(torch.t(A)[j]!=0)
			y[nx]=y[nx]/term
	return y







print("test performance--------------")
task=1
testdata=test(size,task)


#TrainSet
print("training performance--------------")
task=0
testdata=test(size,task)







# quit()
# X = np.array([i for i in range(T)])

# plt.plot(X,epoch_loss_list ,color="g",label='Training Loss')

# plt.xlabel('epoch')
# plt.ylabel('epoch loss')
# plt.legend()
# plt.show()

