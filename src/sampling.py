import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
np.random.seed(0)
from torch.utils.data import DataLoader, Dataset




class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return torch.tensor(image), torch.tensor(label)



def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test


# should return two_dictionaries, one is for train, one is for test
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
"""
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test




def svhn_noniid(dataset, num_users, args):
    """
    Sample non-I.I.D client data from svhn dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = args.num_shards, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.labels[0:len(idxs)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test




def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test


def svhn_non_iid_new(dataset, num_users, args):
    # num_shards, num_imgs = 200, 250
    num_shards, num_imgs = int(args.num_shards), int(60000 / args.num_shards)

    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.dataset.targets)
    dataset_indice = idxs
    labels = np.array([dataset.labels[i] for i in dataset_indice])
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs_label = idxs_labels[1, :]
    dict_users_train_label = {i: np.array([]) for i in range(num_users)}
    dict_users_test_label = {i: np.array([]) for i in range(num_users)}

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            dict_users_train_label[i] = np.concatenate(
                (dict_users_train[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)

            dict_users_test_label[i] = np.concatenate(
                (dict_users_test[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)
    current_max = 0
    for sublist in dict_users_train.values():
        if sublist.max() >= current_max:
            current_max = sublist.max()
    # print(current_max)
    return dict_users_train, dict_users_test

def cifar10_noniid(dataset, num_users, args):
    # num_shards, num_imgs = 200, 250
    num_shards, num_imgs = int(args.num_shards), int(45000 / args.num_shards)

    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.dataset.targets)
    dataset_indice = dataset.indices
    labels = np.array([dataset.dataset.targets[i] for i in dataset_indice])
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs_label = idxs_labels[1, :]
    dict_users_train_label = {i: np.array([]) for i in range(num_users)}
    dict_users_test_label = {i: np.array([]) for i in range(num_users)}

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            dict_users_train_label[i] = np.concatenate(
                (dict_users_train[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)

            dict_users_test_label[i] = np.concatenate(
                (dict_users_test[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)
    current_max = 0
    for sublist in dict_users_train.values():
        if sublist.max() >= current_max:
            current_max = sublist.max()
    # print(current_max)
    return dict_users_train, dict_users_test



def cifar100_non_iid(dataset, num_users, args):
    # num_shards, num_imgs = 200, 250
    num_shards, num_imgs = int(args.num_shards), int(200 * 250 / args.num_shards)

    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs_label = idxs_labels[1, :]
    dict_users_train_label = {i: np.array([]) for i in range(num_users)}
    dict_users_test_label = {i: np.array([]) for i in range(num_users)}

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            dict_users_train_label[i] = np.concatenate(
                (dict_users_train[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)

            dict_users_test_label[i] = np.concatenate(
                (dict_users_test[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)

    return dict_users_train, dict_users_test








# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 250
#     dict_users_train = {i: np.array([]) for i in range(num_users)}
#     dict_users_test = {i: np.array([]) for i in range(num_users)}
#     idx_shard = [i for i in range(num_shards)]
#     idxs = np.arange(num_shards*num_imgs)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(dataset.targets)

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     # idxs_labels is a two dimensional array
#     # the first row is the row index
#     # the second row is the label
#     # like:
#     # [[29513, 16836, 32316, ..., 36910, 21518, 25648],
#     # [    0,     0,     0, ...,     9,     9,     9]]

#     idxs = idxs_labels[0, :]
#     # idxs is the second row of idxs_labels


#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         # rand_set is 200 shard里面选两个
#         idx_shard = list(set(idx_shard) - rand_set)
#         # idx_shard 是除了rand_set之外的剩下的，为下一轮做准备
#         for rand in rand_set:
#             dict_users_train[i] = np.concatenate(
#                 (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)

#             dict_users_test[i] = np.concatenate(
#                 (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
#     return dict_users_train, dict_users_test