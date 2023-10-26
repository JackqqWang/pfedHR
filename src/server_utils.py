import numpy as np
import torch
import torch.nn.functional as F
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
import random
from torch.optim import Adam
import json

from torchvision import transforms

from tools import *
# import args
# key is a string '[CNN, layer index]', value is [total_feat_in, total_feat_out, layer_name, total_feat_in_size, total_feat_out_size]
# output is '[CNN, layer index]': [CNN, layer_name]
from options import args_parser
args = args_parser()
device = args.device

def process_type_list(input_list):
    result_list = []
    for inner_list in input_list:
        converted_list = []
        for pair in inner_list:
            converted_pair = [int(pair[0]), pair[1]]
            converted_list.append(converted_pair)
        result_list.append(converted_list)
    return result_list



class SimDataset(Dataset):
    def __init__(self, original_dataset):
        self.s = 0.5
        self.dataset = original_dataset
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomResizedCrop(32, (0.8, 1.0)),
                                              transforms.Compose(
                                                  [transforms.RandomApply([transforms.ColorJitter(0.8 * self.s,
                                                                                                  0.8 * self.s,
                                                                                                  0.8 * self.s,
                                                                                                  0.2 * self.s)],
                                                                          p=0.8),
                                                   transforms.RandomGrayscale(p=0.2)
                                                   ])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        x = transforms.ToPILImage()(x.to('cpu'))
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return transforms.ToTensor()(x1).cuda(), transforms.ToTensor()(x2).cuda()


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def fine_tune(model, dataloader, loss_func):
    optim = Adam(model.parameters(), 1e-3)
    model.train()
    model.to(device)
    for _ in range(args.sfine):
        for i, data in enumerate(dataloader):
            x, y = data
            optim.zero_grad()
            out = model(x.cuda())
            loss = loss_func(out, y.cuda())
            loss.backward()
            optim.step()
    return model


def fine_tune2(model, dataloader, loss_func):
    optim = Adam(model.parameters(), 1e-3)
    model.train()
    model.to(device)
    for _ in range(args.sfine):
        for i, data in enumerate(dataloader):
            x1, x2 = data
            optim.zero_grad()
            out1 = model(x1.cuda())
            out2 = model(x2.cuda())
            loss = loss_func(out1, out2)
            loss.backward()
            optim.step()
    return model


def compute_sim(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sims = []
    for i, data in enumerate(dataloader):
        x, y = data
        out1 = model1(x.cuda())
        out2 = model2(x.cuda())
        sim = cos(out1, out2)
        sims.append(torch.mean(sim, dim=0))
    sim_avg = torch.mean(torch.stack(sims, dim=0), dim=0)
    return sim_avg


def match_best_model_2(model_pool_with_mlp, local_model_dict, server_train_data):
    dataloader = DataLoader(SimDataset(server_train_data), 8, shuffle=True)
    loss_func = SimCLR_Loss(8)

    print('start fine tuning')
    for cid, cmodel in local_model_dict.items():
        new_model = fine_tune2(cmodel, dataloader, loss_func)
        local_model_dict[cid] = new_model
    for i in range(len(model_pool_with_mlp)):
        new_model = fine_tune2(model_pool_with_mlp[i], dataloader, loss_func)
        model_pool_with_mlp[i] = new_model

    print('start matching')
    output_dict = {}
    for cid, cmodel in local_model_dict.items():
        best_sim = -np.Inf
        best_model = None
        for server_model in model_pool_with_mlp:
            sim = compute_sim(cmodel, server_model, dataloader)
            if sim > best_sim:
                best_sim = sim
                best_model = server_model
        output_dict[cid] = best_model

    return output_dict


def match_best_model(model_pool_with_mlp, local_model_dict, server_train_data):
    dataloader = DataLoader(server_train_data, 8, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    print('start fine tuning')
    for cid, cmodel in local_model_dict.items():
        new_model = fine_tune(cmodel, dataloader, loss_func)
        local_model_dict[cid] = new_model
    for i in range(len(model_pool_with_mlp)):
        new_model = fine_tune(model_pool_with_mlp[i], dataloader, loss_func)
        model_pool_with_mlp[i] = new_model

    print('start matching')
    output_dict = {}
    for cid, cmodel in local_model_dict.items():
        best_sim = -np.Inf
        best_model = None
        for server_model in model_pool_with_mlp:
            sim = compute_sim(cmodel, server_model, dataloader)
            if sim > best_sim:
                best_sim = sim
                best_model = server_model
        output_dict[cid] = best_model

    return output_dict

def comb_with_mlp(candidate_model_combine, local_model_dict, prepare_layer_size_dict, input_size):
# candidate_model_combine is a list [[client index,'layer name'],...]
# local_model_dict is a list, key is client index, value is a model
    model_pool_with_mlp = []
    for layers in candidate_model_combine:
        network_layers = []
        layer_sizes = []
        for layer in layers:
            client_id, layername = layer[0], layer[1]
            network_layers.append(getattr(local_model_dict[int(client_id)], layername))
            layer_sizes.append(prepare_layer_size_dict["['" + str(client_id) + "', '" + layername + "']"])
        model_pool_with_mlp.append(Combined_model(network_layers, layer_sizes, input_size).cuda())
    return model_pool_with_mlp


def sort_layers(input_layers):

    for i in range(len(input_layers)):
        for j in range(len(input_layers) - i - 1):
            if input_layers[j][1][-1] > input_layers[j + 1][1][-1]:
                input_layers[j], input_layers[j + 1] = input_layers[j + 1], input_layers[j]
    return input_layers

def sample_models(input_cluster, min_layers, max_layers, expected_num_models):
    output_models = []
    num_clusters = len(input_cluster.keys())
    while True:
        for i in range(min_layers, max_layers+1):
            sampled_layers = []
            layer_list = [layer for layers in input_cluster.values() for layer in layers]
            for j in range(num_clusters):
                layer = random.sample(input_cluster[j], 1)[0]
                sampled_layers.append(layer)
                layer_list.remove(layer)
            sampled_layers = sampled_layers + random.sample(layer_list, i - num_clusters)
            conv_layers = []
            fc_layers = []
            for sampled_layer in sampled_layers:
                if sampled_layer[1][0:-1] == 'conv':
                    conv_layers.append(sampled_layer)
                elif sampled_layer[1][0:-1] == 'fc':
                    fc_layers.append(sampled_layer)
                else:
                    print('invalid layer, pass')
            if len(conv_layers) == 0:
                break
            if len(fc_layers) == 0:
                break
            conv_layers = sort_layers(conv_layers)
            fc_layers = sort_layers(fc_layers)
            output_models.append(conv_layers + fc_layers)
            if len(output_models) >= expected_num_models:
                break
        if len(output_models)>= expected_num_models:
            return output_models[0:expected_num_models]






def process_from_index_to_name(input, reference):
    # input {0:[['CNN2', 2], ['CNN2', 3], ['CNN2', 4]], 1:[['CNN2', 1], ['CNN3', 2], ['CNN2', 4]]...}
    # reference: key is a string '[CNN, layer index]', value is [CNN, layer name]
    # output {0:[['CNN2', layer_name],['CNN3', layer_name],['CNN1', layer_name] ]}
    input_transfer = {}
    result = {}
    for key, value in input.items():
        temp = []
        for item in value:
            temp.append(str(item))
        input_transfer[key] = temp
    for key,value in input_transfer.items():
        temp = []
        for item in value:
            temp.append(reference[item])
        result[key] = temp
    return result



def process_format(input_dic):
    output_dic = {}
    for key, value in input_dic.items():
        res = key.strip('][').split(', ')
        temp_key = [res[0], value[2]]
        output_dic[key] = temp_key
    return output_dic





def create_mlp(input_dim, output_dim, hidden_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    return model



def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def layer_feature(idxs_users, model_dict, dataset, client_model_info, args):
    # client_model_info key is client index, value is 'CNN1' 'CNN2'... string
    # model_dict key is idx, value is a model
    def hook_fn_forward(module, input, output):
        total_feat_in.append(input)
        total_feat_out.append(output)
        total_feat_in_size.append(input[0].shape)
        total_feat_out_size.append(output[0].shape)
        return None
    trainloader = torch.utils.data.DataLoader(dataset, batch_size = len(dataset), shuffle = True, num_workers = 5)
    result_dict = {}
    for idx in idxs_users:
        current_model = model_dict[idx]
        total_feat_in = []
        total_feat_out = []
        total_feat_in_size = []
        total_feat_out_size = []
        with torch.no_grad():
            current_model.eval()
            current_model.to(args.device)
            handle_list = []
            # layer_index = 0
            layer_name_list = []

            for name, layer in current_model.named_children():
                
                handle = layer.register_forward_hook(hook_fn_forward)
                handle_list.append(handle)
                # if has_numbers(name):
                layer_name_list.append(name)

            for images, _ in trainloader:
                images = images.to(args.device)
                outputs = current_model(images)
            for handle in handle_list:
                handle.remove()
            layer_index_name_dict = dict(zip(range(len(layer_name_list)), layer_name_list))

            for layer_id in range(len(total_feat_in)):
                current_key = [idx, layer_id] # model name, layer id
                current_key = str(current_key)
                result_dict[current_key] = [total_feat_in[layer_id], total_feat_out[layer_id], layer_index_name_dict[layer_id],total_feat_in_size[layer_id], total_feat_out_size[layer_id]]
                
    return result_dict

def extract_embedding(input_dict):
    out_embedding_only = {}
    for key, value in input_dict.items():
        out_embedding_only[key] = value[0:2] 
    return out_embedding_only

def extrac_size(input_dict):
    out_embedding_size_only = {}
    for key, value in input_dict.items():
        out_embedding_size_only[key] = value[-2:] 
    return out_embedding_size_only

def embedding_process(server_output_dict):
    # input is a dictionary: key is a string '[CNN, layer index]', value is [total_feat_in, total_feat_out, infeature_size, out_feat_size]
    # output is get the same dimension for input and the same dimension for output

    squeeze_dict = {}
    for key,value in server_output_dict.items():
        value_temp = []
        for index in range(len(value[0:2])):
            emb = value[index]
            if index == 0:
                if len(list(emb[0].size())) != 2:
                    emb_temp = F.adaptive_avg_pool2d(emb[0], (1, 1)).squeeze()
            if index == 1:
                if len(list(emb.size())) != 2:
                    emb_temp = F.adaptive_avg_pool2d(emb, (1, 1)).squeeze()
            value_temp.append(emb_temp)
        squeeze_dict[key] = value_temp
    # squeeze_dict, in and out feature emb has been squeezed to [10,n]
    squeeze_feature_in_size_list = []
    squeez_feature_out_size_list = []
    for value in squeeze_dict.values():
        squeeze_feature_in_size_list.append(value[0].size(-1))
        squeez_feature_out_size_list.append(value[1].size(-1))
    max_in_features_size = max(squeeze_feature_in_size_list)
    max_out_featues_size = max(squeez_feature_out_size_list)
    result_dict = {}
    for key, value in squeeze_dict.items():
        value_temp = []
        p1d = (0, max_in_features_size - value[0].size(-1))
        p2d = (0, max_out_featues_size - value[1].size(-1))
        if value[0].size(-1) < max_in_features_size:
            value[0] = F.pad(value[0], p1d, "constant", 1)
        if value[1].size(-1) < max_out_featues_size:
            value[1] = F.pad(value[1], p2d, "constant", 1)
        value_temp.append(value[0].detach().cpu())
        value_temp.append(value[1].detach().cpu())
        result_dict[key] = value_temp
    return result_dict

# def candidate_model_generator(input_dict):

#     # e.g. layer_cluster_result = {0: [[CNN1, 0],[CNN2, 1],[CNN2, 4]]
#         #                              1: [[CNN3, 2],[CNN2, 2]]
#         #                              2: [[CNN3, 1],[CNN2, 3]]}

#         # condition 1: the samples must cover all the clusters
#         # condition 2: follow the layer order
#         # return a list of candidate models
#     # return a dictionary:
#     # key is the candidate id, value is a list [[CNN1, 0], [CNN2, 2]...]

#     return result_dict

from itertools import product, combinations

def find_combinations(data):
    # get a list of sublists for each cluster
    new_data = {}
    for key, value in data.items():
        if len(value)!=0:
            new_data[key] = value
    

    cluster_lists = list(new_data.values())
    
    # create a list of all possible combinations of sublists, with at least one sublist per cluster
    all_combinations = []
    for num_sublists in range(len(cluster_lists)):
        for comb in combinations(cluster_lists, num_sublists+1):
            for prod in product(*comb):
                all_combinations.append(list(prod))
    
    # filter out combinations where the second item in all sublists is the same
    filtered_combinations = []
    for comb in all_combinations:
        if len(set([x[1] for x in comb])) == len(new_data):
            filtered_combinations.append(comb)
    
    # sort each combination based on the second item in each sublist
    sorted_combinations = []
    for comb in filtered_combinations:
        sorted_comb = sorted(comb, key=lambda x: x[1])
        sorted_combinations.append(sorted_comb)
    
    # create a dictionary with the sorted combinations as values and an index as the key
    result = {}
    for i, comb in enumerate(sorted_combinations):
        result[i] = comb
    
    return result

