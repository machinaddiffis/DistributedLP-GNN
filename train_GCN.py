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
parser.add_argument('-K','--K',type=int,default=1)
parser.add_argument('-lr','--lr',type=float,default=1e-3)
parser.add_argument('-size','--size',type=int,default=1000)#radius
parser.add_argument('-batch','--batch',type=int,default=1)#mode
parser.add_argument('-num','--num',type=int,default=1)#mode
parser.add_argument('-epoch','--epoch',type=int,default=1000)
parser.add_argument('-L','--L',type=int,default=5)
args = parser.parse_args()
#inference
training_data_len=args.num
best_acc = 1e20  # best test accuracy
lr=args.lr
batch=args.batch
size=args.size
K=args.K
L=args.L




if torch.cuda.is_available():
	device = "cuda"
	
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

print('==> Building model..')


net=GCN(2,2,8).to(device)
#optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer.zero_grad()

def train(max_epoch):
	epoch_loss_list=[]
	rp_list=[]
	rd_list=[]
	for epoch in range(max_epoch):
		train_loss=0
		rp=[]
		rd=[]
		for iter in range(training_data_len):
			with open(f'./instance/size_{size}/LPinstance_{size}_{iter}.pkl', 'rb') as f:
				data_list = pickle.load(f)
			A=torch.tensor(data_list[0],dtype=torch.float32).to(device)
			M,N=A.shape
			x=torch.zeros(size=(N,2))
			y=torch.zeros(size=(M,2))
				 
			x = torch.as_tensor(x,dtype=torch.float32).to(device)
			y = torch.as_tensor(y,dtype=torch.float32).to(device)
		
			pred_primal,pred_dual=net(A,x,y)
			pred_primal=pred_primal.squeeze(1)
			pred_dual=pred_dual.squeeze(1)
			primal=torch.tensor(data_list[1],dtype=torch.float32).to(device)
			dual=torch.tensor(data_list[2],dtype=torch.float32).to(device)

			loss = criterion(pred_primal, primal)+criterion(pred_dual,dual)
			rp.append(abs(torch.sum(pred_primal)-torch.sum(primal))/torch.sum(primal))
			rd.append(abs(torch.sum(pred_dual)-torch.sum(dual))/torch.sum(dual))
			if iter %batch==0 :
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
			train_loss+=loss.item()
		train_loss/=training_data_len
		if train_loss<best_acc:
			torch.save(net.state_dict(), f'./pretrain/best_GCN_size{M}_num{training_data_len}_L{L}.pth')
		epoch_loss_list.append(train_loss)
		rp_list.append(torch.mean(torch.tensor(rp)))
		rd_list.append(torch.mean(torch.tensor(rd)))
		print(f"-------------------current epoch:{epoch}/{max_epoch},     epoch loss:{train_loss},   RP: {torch.mean(torch.tensor(rp))}-------------------")
		data = [epoch_loss_list,rp_list,rd_list]
		with open(f'./log/GCNloss_size{size}_num{training_data_len}_L{L}.pkl', 'wb') as f:
			pickle.dump(data, f)
	return epoch_loss_list

max_epoch=args.epoch
epoch_loss_list=train(max_epoch)

