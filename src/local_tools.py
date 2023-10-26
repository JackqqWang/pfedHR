import torch
from tools import *
from options import args_parser
args = args_parser()


def model_assign_generator(client_number, args=args):
    # should be the 5*
    result_dict = {}
    client_id_model_name = {}
    client_num_each_model = int(client_number/4)
    # print(client_num_each_model)
    model_list = ['CNN1', 'CNN2', 'CNN3', 'CNN4']
    for i in range(4):
        for j in range(client_num_each_model):
            # print(i+j)
            if model_list[i] == 'CNN1':
                result_dict[i*client_num_each_model+j] = CNN1(args=args)
                client_id_model_name[i*client_num_each_model+j] = 'CNN1'
            elif model_list[i] == 'CNN2':
                result_dict[i*client_num_each_model+j] = CNN2(args=args)
                client_id_model_name[i*client_num_each_model+j] = 'CNN2'
            elif model_list[i] == 'CNN3':
                result_dict[i*client_num_each_model+j] = CNN3(args=args)
                client_id_model_name[i*client_num_each_model+j] = 'CNN3'
            elif model_list[i] == 'CNN4':
                result_dict[i*client_num_each_model+j] = CNN4(args=args)
                client_id_model_name[i*client_num_each_model+j] = 'CNN4'
            # elif model_list[i] == 'E':
            #     result_dict[i*client_num_each_model+j] = CNN5(args=args)
            else:
                print("no correct model loaded, exit!")  
                exit()        
    return result_dict, client_id_model_name
