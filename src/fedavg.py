import os
from sklearn import cluster
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
import time
from models import *
from options import args_parser
from utils import average_weights, exp_details, get_datasets
from update import LocalUpdate
from test import test_inference
from tools import *
from sampling import DatasetSplit


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')


    args = args_parser()
    exp_details(args)

    device = args.device

    # load dataset and user groups
    # train_dataset, test_dataset, dict_users_train, dict_users_test = get_datasets(args)
    client_train_dataset, test_dataset, dict_users_train, dict_users_test, server_train_data = get_datasets(args)

    # BUILD MODEL
    if args.model_same:
        if args.model == 'A':
            global_model = CNN1(args=args)
        elif args.model == 'B':
            global_model = CNN2(args=args)
        elif args.model == 'C':
            global_model = CNN3(args=args)
        elif args.model == 'D':
            global_model = CNN4(args=args)           
        elif args.model == 'E':
            global_model = CNN5(args=args)
        else:
            exit('Error: unrecognized model')


        
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    avg_local_test_losses_list, avg_local_test_accuracy_list = [],[]
    avg_local_train_losses_list = []
    print_every = 1

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_train_losses = [], []
        local_test_losses, local_test_accuracy = [], []
        print(f'\n | Communication Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:

            local_model = LocalUpdate(args = args, dataset = client_train_dataset, idxs = dict_users_train[idx])
            w, loss = local_model.update_weights(model = copy.deepcopy(global_model), global_round=epoch)
            trained_local_model = copy.deepcopy(global_model)
            trained_local_model.load_state_dict(w)

            if args.customize_test:

            ################## each client has each test #################################
                test_loader_for_each_client = torch.utils.data.DataLoader(
                    dataset=DatasetSplit(client_train_dataset, dict_users_test[idx]),
                    shuffle=True,
                )
                test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)
            ####################################################################################
            else:
                test_loader_for_share = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                shuffle=True   
                )
                test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_share)



            
            local_weights.append(copy.deepcopy(w))
            local_train_losses.append(copy.deepcopy(loss.item()))
            local_test_losses.append(test_loss)
            local_test_accuracy.append(test_acc)

        loss_avg_train_loss = sum(local_train_losses) / len(local_train_losses)
        avg_local_train_losses_list.append(loss_avg_train_loss)

        loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        avg_local_test_losses_list.append(loss_avg_test_loss)
        loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        avg_local_test_accuracy_list.append(loss_avg_test_accuracy)
        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Stats after {epoch+1} global rounds:')
            print(f'Local Avg Training Loss : {loss_avg_train_loss}')
            print(f'Local Avg Test Loss : {loss_avg_test_loss}')
            print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
    print("**********")
    print("last 10 round avg test accuracy is {}".format(sum(avg_local_test_accuracy_list[-10:])/len(avg_local_test_accuracy_list[-10:])))

    save_path = './fedavg_save/fedavg_{}_{}_{}_iid[{}]_E[{}]_cus_test_{}_frac_{}_shards_{}/'.format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.customize_test, args.frac, args.num_shards)
# Check whether the specified path exists or not
    isExist = os.path.exists(save_path)

    if not isExist:
  # Create a new directory because it does not exist 
        os.makedirs(save_path)
        print("The new directory is created!")
    with open(save_path + 'local_avg_train_loss.txt', 'w') as filehandle:
        for listitem in avg_local_train_losses_list:
            filehandle.write('%s\n' % listitem)

    with open(save_path + 'local_avg_test_losses_list.txt', 'w') as filehandle:
        for listitem in avg_local_test_losses_list:
            filehandle.write('%s\n' % listitem) 

    with open(save_path + 'local_avg_test_accuracy_list.txt', 'w') as filehandle:
        for listitem in avg_local_test_accuracy_list:
            filehandle.write('%s\n' % listitem) 
    
    matplotlib.use('Agg')

        # Plot Loss curve
    plt.figure()
    plt.title('Local Average Training Loss vs Communication rounds')
    plt.plot(range(len(avg_local_train_losses_list)), avg_local_train_losses_list, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_train_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))

    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Local Average Test Loss vs Communication rounds')
    plt.plot(range(len(avg_local_test_losses_list)), avg_local_test_losses_list, color='k')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_test_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))

    plt.figure()
    plt.title('Local Average Test Accuracy vs Communication rounds')
    plt.plot(range(len(avg_local_test_accuracy_list)), avg_local_test_accuracy_list, color='r')
    plt.ylabel('Test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_test_accuracy.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))

