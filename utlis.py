import numpy
import torch

#from LP problem to Graph

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
