# -*- coding:utf-8 -*-

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter, FMCAargs
from model import FMCADTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.EarlyStoping import EarlyStopping
from utils.TestModel import test_model
from utils.ShowResult import show_result

from dataset import *
from FMCAutils import *
from copy import deepcopy
from torch.autograd import Variable
import pandas as pd
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    dataset_name = DATASET 
    split_random = True
    dataset = dataset_config[dataset_name]
    root_path = dataset_config[dataset_name]

    input_path = root_path + "/"
    output_path = root_path + "/" + "output"

    decompose = "bcm"
    decompose_protein = "category"


    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    datasetSmiles, datasetProtein, datasetLabel = load_all_dataset(input_path)
    train_data_list, test_data_list = split_train_test_set(datasetSmiles, datasetProtein, datasetLabel)


    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(
            i_fold, train_data_list, k=K_Fold)

        train_dataset = [item.split(',') for item in train_dataset]
        valid_dataset = [item.split(',') for item in valid_dataset]
        test_dataset = [item.split(',') for item in test_data_list]
        trainSmiles, trainProtein, trainLabel, \
            valSmiles, valProtein, valLabel, \
            testSmiles, testProtein, testLabel, \
            frag_set_d, frag_set_p, \
            frag_len_d, frag_len_p, \
            words2idx_d, words2idx_p = load_frag(train_dataset, valid_dataset, test_dataset, decompose,
                                                               decompose_protein, unseen_smiles=False, k=3,
                                                               split_random=split_random)
        args = FMCAargs()
        n=3
        args['max_drug_seq'] = max(frag_len_d)
        args['max_protein_seq'] = max(frag_len_p)
        args['input_d_dim'] = len(frag_set_d) + 1
        args['input_p_dim'] = len(frag_set_p) + 1
        args['d_channel_size'][n - 1][0] = args['max_drug_seq']
        args['p_channel_size'][n - 1][0] = args['max_protein_seq']
        args['d_channel_size'] = args['d_channel_size'][n - 1]
        args['p_channel_size'] = args['p_channel_size'][n - 1]

        trainDataset = NewDataset(trainSmiles, trainProtein, trainLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                                  args['max_protein_seq'])
        validDataset = NewDataset(valSmiles, valProtein, valLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                                  args['max_protein_seq'])
        testDataset = NewDataset(testSmiles, testProtein, testLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                                 args['max_protein_seq'])
        train_size = len(train_dataset)

        train_dataset_loader = DataLoader(dataset=trainDataset, batch_size=hp.Batch_size, shuffle=True,
                                        drop_last=True)

        valid_dataset_loader = DataLoader(dataset=validDataset, batch_size=hp.Batch_size, shuffle=False,
                                          drop_last=True)
        test_dataset_loader = DataLoader(dataset=testDataset, batch_size=hp.Batch_size, shuffle=False,
                                         drop_last=True)

        """ create model"""
        model = MODEL(hp,args).to(DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)

        Loss = torch.nn.BCEWithLogitsLoss()

        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_compounds, train_proteins, train_labels = train_data
                train_compounds = train_compounds.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_labels = train_labels.float().to(DEVICE)

                predicted_interaction = model(train_compounds, train_proteins)
                predicted_interaction = torch.squeeze(predicted_interaction)
                
                train_loss = Loss(predicted_interaction, train_labels)
                train_losses_in_epoch.append(train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  

            """valid"""
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:

                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.float().to(DEVICE)

                    valid_scores = model(valid_compounds, valid_proteins)
                    valid_scores = torch.squeeze(valid_scores)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = valid_scores.to('cpu').data.numpy()
                    threshold = 0.5
                    valid_predictions = [1 if i else 0 for i in (valid_scores >= threshold)]
                    valid_scores = valid_scores.flatten().tolist()

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _, = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        validset_test_stable_results, _, _, _, _, _, = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)

        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)

if __name__ == '__main__':
    run_model(SEED=114514, DATASET='BIOSNAP',
              MODEL=FMCADTI, K_Fold=5, LOSS='Loss')
