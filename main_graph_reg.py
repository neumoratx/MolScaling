import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from splitter import random_split, scaffold_split


from datasets.molnet import MoleculeDataset
from model.gnn import GNN
from model.mlp import MLP

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return

def get_num_task(dataset):
    # Get output dimensions of different tasks
    if dataset == 'mdck':
        return 1
    
def compute_mean_mad(values):
    meann = torch.mean(values)
    mad = torch.std(values)
    return meann, mad

def train_general(model, device, loader, optimizer):
    model.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        pred = output_layer(h)
        
        y = batch.y.view(pred.shape).float()
        y = ((y-meann)/mad)
        loss = reg_criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            h = global_mean_pool(model(batch), batch.batch)
            pred = output_layer(h)
    
        true = batch.y.view(pred.shape).float()
        y_true.append(true)
        y_pred.append(pred)


    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = (torch.cat(y_pred, dim=0)*mad + meann).cpu().numpy()

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r = np.corrcoef(y_true.T, y_pred.T)[1,0]
    return {'RMSE': rmse, 'MAE': mae, 'pearson_R': pearson_r}, y_true, y_pred    
if __name__ == '__main__':
    
    # Set a MLflow Experiment
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("MLflow GNN")
    
    # Set hyperparameters
    dataset_name = 'mdck'
    num_tasks = get_num_task(dataset_name)
    split = 'random'
    batch_size = 256
    num_layer = 5
    emb_dim = 300
    dropout_ratio = 0.5
    lr = 1e-3
    epochs = 20
    
    # Log params
    params = {
        'dataset_name': dataset_name,
        'split': split,
        'batch_size': batch_size,
        'num_layer': num_layer,
        'emb_dim': emb_dim,
        'dropout_ratio': dropout_ratio,
        'lr': lr,
        'epochs': epochs
    }
    mlflow.log_params(params)
    
    # Set your dataset directory
    dataset_folder = '/home/ubuntu/adme/MolScaling/datasets/molecule_net/'
    dataset = MoleculeDataset(dataset_folder + dataset_name, dataset=dataset_name)
    
    # Set device and seed
    device = torch.device('cuda') 
    seed = 0
    seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    

    # Initalize model
    model_param_group = []
    model = GNN(num_layer=num_layer, emb_dim=emb_dim, drop_ratio=dropout_ratio).to(device)
    output_layer = MLP(in_channels=emb_dim, hidden_channels=emb_dim, 
                        out_channels=num_tasks, num_layers=1, dropout=0).to(device)

    model_param_group.append({'params': output_layer.parameters(),'lr': lr})
    model_param_group.append({'params': model.parameters(), 'lr': lr})
    print(model)                

    # Initalize optimizer and metrics
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=0)
    reg_criterion = torch.nn.MSELoss()
    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE', 'pearson_R']
    best_val_mae, best_val_idx = 1e10, 0
    
    # Split data
    if split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif split == 'random':
        smiles_list = pd.read_csv(dataset_folder + dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles),_ = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=seed, smiles_list=smiles_list)
        print('randomly split')

    # Set dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8)

    meann, mad = compute_mean_mad(train_dataset.data.y)
    train_func = train_general
    eval_func = eval_general

    with mlflow.start_run():
        for epoch in range(1, epochs + 1):
            loss_acc = train_func(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            train_result, train_target, train_pred = eval_func(model, device, train_loader)
            
            val_result, val_target, val_pred = eval_func(model, device, val_loader)
            test_result, test_target, test_pred = eval_func(model, device, test_loader)
            test_result_list.append(test_result)
            
            train_result_list.append(train_result)
            val_result_list.append(val_result)

            for metric in metric_list:
                print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))

            if val_result['MAE'] < best_val_mae:
                best_val_mae = val_result['MAE']
                best_val_idx = epoch - 1
                
            # mlflow.log_metric("r2_score", train_result['pearson_R'], step=epoch)
            mlflow.log_metrics(val_result, step=epoch)

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))