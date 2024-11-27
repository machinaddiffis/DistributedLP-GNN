# DistributedLP-GNN
Code implementation for DistributedLP-GNN

###Requirements

## Requirements

Linux

–python 3.8.13

–pytorch 1.10.2

–cudatoolkit 11.3

–ortools  9.11.4210


## Repository structure


```

instance

​	–size_100                //training and evaluation instances

​	–size_500                //training and evaluation instances

  –size_1000               //training and evaluation instances

  –size_1500               //training and evaluation instances 

log                    //all test logs

pretrain                // training model folder will be saved here
```
## Data generation
Run the following commands to generate different instances and the corresponding solutions

```
python gen_solution.py
```

## Train 

Run the following commands to train models of different sizes
```
python train_DLPGNN.py
python train_GCN.py
```

## Test

Run the following command to test the pre-trained model

```
python test_DLPGNN.py
python test_GCN.py
```
all logs will be saved in log folder.
