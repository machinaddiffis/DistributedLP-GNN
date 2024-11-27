from model import *
import torch 
import gzip
import pickle
import os
import random
import torch
import torch.optim as optim
from model import DLPGNN
import argparse
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-size','--size',type=int,default=100)#radius
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


net=DLPGNN(L=L,K=16,M=size,N=size).to(device)
model_name=f"best_DLPGNN_size{size}_num100_L5.pth"
net.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
print(model_name)
net.eval()

#optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer.zero_grad()
criterion = nn.MSELoss()
def test(size,task):
	R_primal=[]
	R_dual=[]
	train_loss=0
	if task==1:# test set
		training_data_len=100
		pointer=500
	else: # train set
		training_data_len=100
		pointer=0

	for iter in range(training_data_len):
		iter=iter+pointer
		with open(f'./instance/size_{size}/LPinstance_{size}_{iter}.pkl', 'rb') as f:
			data_list = pickle.load(f)
		# print(f"{iter}------------------------")
		A=torch.tensor(data_list[0],dtype=torch.float32).to(device)
		pred_primal,pred_dual=net(A)
		primal=torch.tensor(data_list[1],dtype=torch.float32).to(device)
		dual=torch.tensor(data_list[2],dtype=torch.float32).to(device)
		x_dot=resortation_1(A,pred_primal)
		y_dot=resortation_2(A,pred_dual)
		rp=abs(torch.sum(x_dot)-torch.sum(primal))/torch.sum(primal)
		rd=abs(torch.sum(y_dot)-torch.sum(dual))/torch.sum(dual)
	
		R_primal.append(rp)
		R_dual.append(rd)

	R_primal=torch.mean(torch.tensor(R_primal))
	R_dual=torch.mean(torch.tensor(R_dual))
	if task==1:
		print(f"test RP:{R_primal},test RD:{R_dual}")
	else:
		print(f"testing RP:{R_primal},training RD:{R_dual}")
	return R_primal,R_dual

def resortation_1(A,x):
	M,N=A.shape
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
	eps=1e-5
	
	y=torch.max(ones*eps,torch.min(ones,y))
	for j in range(N):
		term=torch.sum(torch.t(A)[j]*y)
		if term <=1:
			nx=torch.where(torch.t(A)[j]!=0)
			y[nx]=y[nx]/term
			# if term==0:
			# 	return None
	return y



print("testing--------------")
task=1
testdata=test(size,task)
#TrainSet

print("training--------------")
task=0
testdata=test(size,task)







# quit()
# X = np.array([i for i in range(T)])

# plt.plot(X,epoch_loss_list ,color="g",label='Training Loss')

# plt.xlabel('epoch')
# plt.ylabel('epoch loss')
# plt.legend()
# plt.show()

