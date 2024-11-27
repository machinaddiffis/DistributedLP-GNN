import numpy as np
from ortools.linear_solver import pywraplp
import pickle
from ortools.linear_solver.python import model_builder
import math
import torch
import os
#params


def problem_gen_sol(seed=0,M=500,N=500):
    list=[]
    #get A
    np.random.seed(seed)
    nums=10*(M+N)
    A = np.zeros((M, N), dtype=int)
    positions = np.random.choice(M * N, nums, replace=False)
    A.ravel()[positions] = 1.0

    list.append(A)

    #def a prob
    #primal
    c_primal=np.ones(N)
    b_primal=np.ones(N)
    # print(C)
    x,obj=solve_lp(A,b_primal,c_primal)
    list.append(x)

    

    # #dual
    c_dual=-np.ones(N)
    b_dual=-np.ones(N)
    A_dual=-A

    x,obj=solve_lp(A_dual,b_dual,c_dual)
    list.append(x)
    x=np.expand_dims(x,axis=1)

    # Feature=SparseLP2Graph(A,b_primal,c_primal)

    return list


def solve_lp(A, b, c):
    M, N = A.shape  #

 
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    if not solver:
        print('无法创建求解器！')
        return None

    # 
    x = []
    for i in range(N):
        x.append(solver.NumVar(0.0, solver.infinity(), f'x{i}'))

    #  c^T * x
    objective = solver.Objective()
    for i in range(N):
      
        objective.SetCoefficient(x[i], float(c[i]))
 
    objective.SetMaximization()

    # Ax <= b
    for i in range(M):
        constraint = solver.Constraint(-solver.infinity(), float(b[i]))  
        for j in range(N):
            constraint.SetCoefficient(x[j], float(A[i, j]))
    
    #get a model
    solver.Solve()

    solution = [x[i].solution_value() for i in range(N)]
    objective_value = objective.Value()
    print(f"Optimal value: {objective_value}")
    # print(f"Optimal solution: {solution}")



    return solution, objective_value




    

if __name__ == "__main__":

    M=1000
    N=M
    testing=False
    folder_path = f"size_{M}"

    test_num=2
    for iter in range(test_num):
        if testing:
            iter=iter+100
        data_list=problem_gen_sol(seed=iter,M=M,N=N)
        data_list.append(iter)
        with open(f'./instance/{folder_path}/LPinstance_{M}_{iter}.pkl', 'wb') as f:
            pickle.dump(data_list, f)
