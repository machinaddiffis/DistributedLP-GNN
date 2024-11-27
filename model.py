import torch
import torch.nn as nn
from utlis import max_product

if torch.cuda.is_available():
	device = "cuda"
	
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"



class GCN(torch.nn.Module):
    def __init__(self,x_size,y_size,feat_size,n_layer=4):
        super(GCN,self).__init__()
        self.embx = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
        )
        self.emby = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
        )
        self.updates = nn.ModuleList()
        self.updates.append(GCN_layer(feat_size,feat_size,feat_size))
        for indx in range(n_layer):
            self.updates.append(GCN_layer(feat_size,feat_size,feat_size))
            
        self.outlayerX = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,1,bias=True),
            nn.LeakyReLU(),
        )

        self.outlayerY = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,1,bias=True),
            nn.LeakyReLU(),
        )
        
        # def _initialize_weights(m):
        #     for m in self.modules():
        #         if isinstance(m,nn.Linear):
        #             torch.nn.init.xavier_uniform_(m.weight,gain=1)
                    
        # self.outlayer.apply(_initialize_weights)
        # self.embx.apply(_initialize_weights)
        # self.emby.apply(_initialize_weights)
        
        
    def forward(self,A,x,y):
        x = self.embx(x)
        y = self.embx(y)

        for index, layer in enumerate(self.updates):
            x,y = layer(A,x,y)
            
        return self.outlayerX(x),self.outlayerY(x)
    
    
class GCN_layer(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size):
        super(GCN_layer,self).__init__()
        self.feat_size = feat_size
        self.embx = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.emby = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.outlayer = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        
        self.embx2 = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.emby2 = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.outlayer2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
              
    def forward(self,A,x,y):

        Ax = self.embx(torch.matmul(A,x))
        y = self.emby(y) + Ax
        y = self.outlayer(y)
        AT = torch.transpose(A,0,1)
        ATy = self.emby2(torch.matmul(AT,y))
        x = self.embx2(x) + ATy
        x = self.outlayer2(x)
        return x,y



class DLPGNN(torch.nn.Module):
    def __init__(self,K=16,L=5,alpha=2,N=500,q_len=4):
        super(DLPGNN,self).__init__()
        self.K=K
        self.L=L+1
        self.theta_g = nn.Parameter(torch.randn(self.L-1,self.K,3))
        self.theta_b = nn.Parameter(torch.randn(self.L-1,self.K,4))
        self.f=nn.Parameter(torch.tensor(100.0))
        self.h=nn.Parameter(torch.ones(N)).to(device)
        self.alpha=alpha

        self.q_len=q_len
        self.theta_q_left = nn.Parameter(torch.randn(self.q_len,2))
        self.theta_q_right = nn.Parameter(torch.randn(self.q_len,2))
        


    def g_theta(self,x,l):
        item=0
        for k in range(self.K):
            item+=self.theta_g[l,k,0]*torch.sigmoid(self.theta_g[l,k,1]*x+self.theta_g[l,k,2])
        return item
    
    def b_theta(self,x,l):
        item=0
        for k in range(self.K):
            power=torch.pow(self.alpha, -torch.torch.nn.functional.relu(self.theta_b[l,k,3]*(x-self.f)))
            item+=self.theta_b[l,k,0]*torch.sigmoid(self.theta_b[l,k,1]*x+self.theta_b[l,k,2])*power
        return item
    
    def q_left(self,x):
        item=0
        for k in range(self.q_len):
            item+=torch.sigmoid(self.theta_q_left[k,0]*x+self.theta_q_left[k,1])
        return item/4
    
    def q_right(self,x):
        item=0
        for k in range(self.q_len):
            item+=torch.sigmoid(self.theta_q_right[k,0]*x+self.theta_q_right[k,1])
        return item/4
         
        

    def forward(self,A):
        column_sums = A.sum(axis=0)
        A_bar = A / column_sums


        self.r=[None]*self.L
        self.ro=[None]*self.L
        self.ro_t=[None]*self.L
        self.ro_max=[None]*self.L
        self.y_delta=[None]*self.L
        self.y=[None]*self.L
        self.x=[None]*self.L
        self.b=[None]*self.L
        M,N=A.shape
        #initial feature
        self.r[0]=self.f*torch.ones(N).to(device)
        self.b[0]=torch.ones(N).to(device)
        self.ro_t[0]=torch.zeros(N).to(device)
        self.x[0]=torch.zeros(N).to(device)
        self.left_mask=torch.zeros(N).to(device)
        #right
        self.y[0]=torch.zeros(M).to(device)
        self.y_delta[0]=torch.zeros(M).to(device)
        self.ro[0]=torch.zeros(M).to(device)
        self.ro_max[0]=torch.zeros(M).to(device)
        self.right_mask=torch.zeros(M).to(device)
        #degree    
        ones_N=torch.ones(N).to(device)
        ones_M=torch.ones(M).to(device)
        self.right_mask=self.q_right(torch.mv(A,ones_N))
        case1=self.q_left(torch.mv(torch.t(A),ones_M))  
        case2=max_product(torch.t(A),self.right_mask)
        self.left_mask=torch.max(case1, case2)
        for l in range(self.L-1):
            self.ro[l+1]=torch.mv(A, self.b[l]).to(device)*(1-self.right_mask)
   
            self.ro_t[l+1]=max_product(torch.t(A),self.ro[l+1]).to(device)*(1-self.left_mask)
            
            self.ro_max[l+1]=max_product(A,self.ro_t[l+1]).to(device)*(1-self.right_mask)
            self.y_delta[l+1]=self.g_theta(((self.ro[l+1]-self.ro_max[l+1]/self.alpha)).to(device),l)*(1-self.right_mask)  
            self.y[l+1]=(self.y[l]+self.y_delta[l+1])*(1-self.right_mask)  
                        
            self.r[l+1]=(self.r[l]-torch.torch.nn.functional.relu(torch.mv(torch.t(A),self.y_delta[l+1])))*(1-self.left_mask)
            medterm1=torch.mul(self.y_delta[l+1],1.0/self.ro[l+1])
            medterm2=torch.mv(torch.t(A),medterm1)

            self.x[l+1]=(self.x[l]+torch.mul(self.b[l],medterm2))*(1-self.left_mask)
            self.b[l+1]=self.b_theta((self.r[l+1]),l)*(1-self.left_mask)

        pred_primal=self.x[self.L-1]
        term=self.y[self.L-1]
        pred_dual=term+self.right_mask+torch.mv(A_bar,self.h*self.left_mask)
        return pred_primal,pred_dual