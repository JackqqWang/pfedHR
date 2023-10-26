import seaborn as sns
import os
import tqdm
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
# import time
from tools import *
from server import *
from models import *
from tools import *
from server_utils import *
from cluster_alg import *
from options import args_parser
from utils import exp_details, get_datasets, get_pub_datasets,get_public_datasets
from update import LocalUpdate

from test import test_inference
from local_tools import *
from sampling import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    path_project = os.path.abspath('..')
     
    exp_details(args)
    device = args.device
    
    # server_train_data = get_server_train(dataset, args)

    train_dataset, test_dataset, dict_users_train, dict_users_test, server_train_data = get_datasets(args)

    if args.public_dataset == args.dataset:
        data_at_server = server_train_data
    else:
        data_at_server = get_public_datasets(args)
    if args.public_dataset == 'cifar10':
        input_size = 32 * 32 * 3
    elif args.public_dataset == 'mnist':
        input_size = 32 * 32 * 3
    elif args.public_dataset == 'svhn':
        input_size = 32 * 32 * 3
    else:
        print('wrong public dataset name')

    if args.model_same == 1:
        if args.model == 'CNN1':
            model_indicator = 'A'
        elif args.model == 'CNN2':
            model_indicator = 'B'
        elif args.model == 'CNN3':
            model_indicator = 'C'
        elif args.model == 'CNN4':
            model_indicator = 'D'
        else:
            print('wrong model name')
        client_model_dict = {}
        for client_idx in range(args.num_users):
            client_model_dict[client_idx] = model_indicator
    # model_dict, server_train_loss = Server.train_server(args = args, dataset = server_train_data)
    client_model_dict = model_generation(args.num_users) # input is number_users, the output key is the index, value is the model name
    # comment: client_model_dict = {0:'A', 1:'A', 2:'B', 3:'B', .....8:'E',9:'E'}
    local_avg_train_losses_list, local_avg_train_accuracy_list = [],[]
    local_avg_test_losses_list, local_avg_test_acc_list = [], []
    local_avg_test_accuracy_list = []
    print_every = 1
    
    #generate local model dict: key is the idx, value is CNN1 CNN2 CNN3 CNN4 CNN5
    model_assign_dict, client_id_model_name = model_assign_generator(args.num_users)
    # comment: model_assign_dict's key is 0,1,2,3,4,5,6,7,8,9 value is a real model, like a CNN model
    # client_id_model_name is the same as client_model_dict, but the value is the model name, not the model itself
    print("start server and client communication:")
    previous_user_list = []
    current_user_id_modelweights_dict = {}
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []

        # current_user_id_modelweights_dict = {} # key is idx, value is the model
        local_test_losses, local_test_accuracy = [],[]
        print('communication round: {} \n'.format(epoch))

        # global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace = False)
        # previous_user_list = idxs_users
        
        for idx in idxs_users:
            test_loader_for_each_client = torch.utils.data.DataLoader(
                dataset=DatasetSplit(train_dataset, dict_users_test[idx]),
                shuffle=True,
            )
            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_train[idx])

            # if args.customize_test:
            if epoch == 0:
                w, loss = local_model.update_weights(model = copy.deepcopy(model_assign_dict[idx]), global_round = epoch)
                trained_local_model = copy.deepcopy(model_assign_dict[idx])
                trained_local_model.load_state_dict(w)
                trained_local_model.to(device)
                test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)
            
            
            else:
                
                if idx in previous_user_list:
                    #TODO pull the idx model down to the local
                    teacher_model = best_model_dict[idx]
                    currrent_student_model = current_user_id_modelweights_dict[idx]
                    w, loss = local_model.k_distll(student_model = currrent_student_model, teacher_model = teacher_model)
                    trained_local_model = copy.deepcopy(current_user_id_modelweights_dict[idx])
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)

                else:
                    w, loss = local_model.update_weights(model = copy.deepcopy(model_assign_dict[idx]), global_round = epoch)
                    trained_local_model = copy.deepcopy(model_assign_dict[idx])
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)

            local_losses.append(copy.deepcopy(loss.detach().cpu().item()))
            local_test_losses.append(test_loss) # len = user number 
            local_test_accuracy.append(test_acc)
            temp_temp_model = copy.deepcopy(model_assign_dict[idx])
            temp_temp_model.load_state_dict(w)
            # local_model_list.append(temp_temp_model)
            current_user_id_modelweights_dict[idx] = temp_temp_model

        loss_avg = sum(local_losses) / len(local_losses)
        loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        local_avg_test_losses_list.append(loss_avg_test_loss)
        loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        local_avg_test_accuracy_list.append(loss_avg_test_accuracy)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Stats after {epoch+1} global rounds:')
            print(f'Local Avg Training Loss : {loss_avg}')
            print(f'Local Avg Test Loss : {loss_avg_test_loss}')
            print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')
        previous_user_list = idxs_users
        ############################ server part #################################
        # server receives a dictionary, where key is idx, value is a model

        ##### Step 1: feed server data into models #########
        ##### Output is a dictionary - key is a list, and value is a list #########
        ################ [CNN1, layer index]: [input embedding, output embedding, input emd size, output emd size, layer parameter]
        local_model_dict = copy.deepcopy(current_user_id_modelweights_dict)
        # server_training_data = server_train_data
        client_model_info = client_model_dict


        server_output_dict = layer_feature(idxs_users, local_model_dict, server_train_data, client_model_info, args)
        # in server_output_dict, key is a string '[client id, layer index]', value is [total_feat_in, total_feat_out, layer_name, total_feat_in_size, total_feat_out_size]
        model_layer_index_to_model_layer_name = process_format(server_output_dict)
        # in model_layer_index_to_model_layer_name,  key is a string '[client id, layer index]', value is [client id, layer name]
        # client_id_model_name is the same as client_model_dict, but the value is the model name, not the model itself
        prepare_layer_size_dict = {}
        for key, value in server_output_dict.items():
            prepare_layer_size_dict[str(model_layer_index_to_model_layer_name[key])] = value[-2:]
        # prepare_layer_size_dict key is '[client id, layer name]', value is [total_feat_in_size, total_feat_out_size]

        server_output_dict_only_embedding = extract_embedding(server_output_dict)
        server_output_dict_only_size = extrac_size(server_output_dict)
        # in server_output_dict_only_embedding, key is a string '[client id, layer index]', value is [total_feat_in, total_feat_out]
        server_output_dict_same_embedding = embedding_process(server_output_dict)
        

        cluster_results = k_cluster(server_output_dict_same_embedding, args.cluster_num, -5, 10)
        # for value in cluster_results.values():
        #     if len(value) == 0:
        #         cluster_results = k_cluster(server_output_dict_same_embedding, args.cluster_num, -5, 10)
        #         break
                # if len(value) == 0:
                #     cluster_results = k_cluster(server_output_dict_same_embedding, args.cluster_num, -5, 2)
        # in cluster_results, key is a cluster id, value is a list of [CNN, layer index]

        ##### Step 2: assemble layers into a candidate pool  #########
        # layer_cluster_result = copy.deepcopy(cluster_results)
        # layer_cluster_result is a dictionary, key is a cluster id, value is a string
        # e.g. layer_cluster_result = {0: [[client id, 0],[client id, 1],[client id, 4]]
        #                              1: [[client id, 2],[client id, 2]]
        #                              2: [[client id, 1],[client id, 3]]}
        for_find_comb_input = {}
        for key, value in cluster_results.items():
            new_value = []
            for item in value:
                temp_temp_temp = model_layer_index_to_model_layer_name[item]
                new_value.append(temp_temp_temp)
            for_find_comb_input[key] = new_value
    

        # in for_find_comb_input, key is a cluster id
        #                             {0: [[client id, layer_name],[client id, layer_name],[client id, layer_name]]
        #                              1: [[client id, layer_name],[client id, layer_name]]
        #                              2: [[client id, layer_name],[client id, layer_name]]}

        #sample_models(input_cluster, min_layers, max_layers, expected_num_models)
        candidate_model_combine = sample_models(for_find_comb_input, 4, 6, args.expected_num_models)
        # candidate_model_combine is a list [[['client index','layer name'],...]]
        # print(prepare_layer_size_dict)
        # print(getattr(local_model_dict[7], 'conv1')())
        # for v in prepare_layer_size_dict.values():
        #     print(v[0])
        #client_id_model_name is the same as client_model_dict, but the value is the model name, not the model itself
        candidate_model_combine_client_id_to_model_name = process_type_list(candidate_model_combine)
        print(candidate_model_combine_client_id_to_model_name)
        candidate_model_combine_show_model = []
        for sublist in candidate_model_combine_client_id_to_model_name:
            temp_sublist = []
            for item in sublist:
                temp_item = []
                temp_item = [client_id_model_name[item[0]],item[1]]
    
                temp_sublist.append(temp_item)
            candidate_model_combine_show_model.append(temp_sublist)
        print(candidate_model_combine_show_model)

        model_pool_with_mlp = comb_with_mlp(candidate_model_combine, local_model_dict, prepare_layer_size_dict, input_size = input_size)
        # prepare_layer_size_dict key is '[client id,layer name]', value is a [input size, output size]
        # local_model_dict: KEY is client index, VALUE is a model
        # candidate_model_combine: [[client index,'layer name'],...]
        # print(model_pool_with_mlp[8](torch.randn(64, 3, 32, 32).cuda()))

        # for each model in model_pool, we train it and we train the local model as well.
        # we compare the output logits and do matching
        ##### Step 3: for each client, we assign a model to that  #########
        # local_model_dict, key is client index, value is a model
        if args.supervised:
            best_model_dict = match_best_model(model_pool_with_mlp,local_model_dict,server_train_data)
        else:
            best_model_dict = match_best_model_2(model_pool_with_mlp,local_model_dict,server_train_data) # list is the candidate model pool, local_model_dict is previous client model dict,
        # print(best_model_dict)
        # output length is the number of clients
        # output is a dictionary, key is client index, value is the best match model from the candidate pool
        

    save_path = './exp_result/{}_{}_com{}_iid_{}_E_{}_sfine_{}_teacherweight_{}_cluster_{}_frac_{}_user_num_{}/'.format(args.dataset, args.public_dataset, args.epochs,
                       args.iid, args.local_ep, args.sfine, args.alpha, args.cluster_num, args.frac, args.num_users)
# Check whether the specified path exists or not
    isExist = os.path.exists(save_path)

    if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")
    
        
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    
    with open(save_path + 'local_avg_train_loss.txt', 'w') as filehandle:
        for listitem in local_avg_train_losses_list:
            filehandle.write('%s\n' % listitem)

    with open(save_path + 'local_avg_test_losses_list.txt', 'w') as filehandle:
        for listitem in local_avg_test_losses_list:
            filehandle.write('%s\n' % listitem) 

    with open(save_path + 'local_avg_test_accuracy_list.txt', 'w') as filehandle:
        for listitem in local_avg_test_accuracy_list:
            filehandle.write('%s\n' % listitem) 
    print("*******last 3 avg local acc****************")
    print("{}".format(sum(local_avg_test_accuracy_list[-3:])/3))
    print("********************************************")

    # Plot Loss curve
    plt.figure()
    plt.title('Local Average Training Loss vs Communication rounds')
    plt.plot(range(len(local_avg_train_losses_list)), local_avg_train_losses_list, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_train_loss.png')
    
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Local Average Test Loss vs Communication rounds')
    plt.plot(range(len(local_avg_test_losses_list)), local_avg_test_losses_list, color='k')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_test_loss.png')

    plt.figure()
    plt.title('Local Average Test Accuracy vs Communication rounds')
    plt.plot(range(len(local_avg_test_accuracy_list)), local_avg_test_accuracy_list, color='r')
    plt.ylabel('Test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_test_accuracy.png')


        
        












