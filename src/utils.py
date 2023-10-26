

import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import torch
# from sklearn.cluster import KMeans
import numpy as np
from torchvision import datasets, transforms
from sampling import *
from options import args_parser
from torch.utils.data import Subset

def average_weights(w):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

def get_pub_datasets(args):
    if args.public_dataset == 'cifar100':
        data_dir = '../data/cifar100'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]
        )
        train_dataset = datasets.CIFAR100(data_dir, train= True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train = False, download= True, transforms = apply_transform)
        if args.iid:
            dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
        else:
            dict_users_train, dict_users_test = cifar100_non_iid(train_dataset, args.num_users, args)


    return train_dataset, test_dataset, dict_users_train, dict_users_test


def get_server_client_train_data(training_dataset):

    ds = training_dataset
    indices = [[] for _ in range(10)]
    for i in range(len(training_dataset)):
        current_label = ds[i][1]
        indices[current_label].append(i)
    sample_indices = [item[:int(0.1 * len(item))] for item in indices]
    client_total_indices = []
    for i in range(len(sample_indices)):
        client_total_indices.append(list(set(indices[i]).difference(set(sample_indices[i]))))
    sample_indices_list = []
    client_total_indices_list = []
    for item in sample_indices:
        sample_indices_list = sample_indices_list + item
    for item in client_total_indices:
        client_total_indices_list = client_total_indices_list + item
    server_train_data = Subset(ds, sample_indices_list)
    client_train_data = Subset(ds, client_total_indices_list)

    return server_train_data, client_train_data









def get_public_datasets(args):

    if args.public_dataset == 'cifar10':
        data_dir = '../data/cifar10'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
        train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)
        server_train_data, _ = get_server_client_train_data(train_dataset)


    if args.public_dataset == 'svhn':
        data_dir = '../data/svhn'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
        train_dataset = datasets.SVHN(data_dir, split = 'train', download = True, transform = apply_transform)
        server_train_data, _ = get_server_client_train_data(train_dataset)


    elif args.public_dataset == 'mnist':
        data_dir = '../data/mnist'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ]
        )

        train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)
        server_train_data, _ = get_server_client_train_data(train_dataset)

    return server_train_data



def get_datasets(args):

    if args.dataset == 'cifar10':
        data_dir = '../data/cifar10'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )

        train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train = False, download = True, transform = apply_transform)

        server_train_data, client_train_data = get_server_client_train_data(train_dataset)
        

        if args.iid:
            dict_users_train,  dict_users_test = iid(client_train_data, args.num_users)
        else:
            dict_users_train, dict_users_test = cifar10_noniid(client_train_data, args.num_users, args)


    if args.dataset == 'svhn':
        data_dir = '../data/svhn'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
        train_dataset = datasets.SVHN(data_dir, split = 'train', download = True, transform = apply_transform)
        test_dataset = datasets.SVHN(data_dir, split = 'test', download = True, transform = apply_transform)
        # server_train_data = get_server_client_train_data(train_dataset)
        server_train_data, client_train_data = get_server_client_train_data(train_dataset)
        if args.iid:
            dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
            # dict_users_train,  dict_users_test = svhn_iid(train_dataset, args.num_users)
        else:
            dict_users_train, dict_users_test = svhn_non_iid_new(train_dataset, args.num_users, args)

    elif args.dataset == 'mnist':

        data_dir = '../data/mnist'


        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ]
        )

        train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)
        test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)
        # server_train_data = get_server_client_train_data(train_dataset)
        server_train_data, client_train_data = get_server_client_train_data(train_dataset)

        if args.iid:
            dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
        else:
            dict_users_train, dict_users_test = mnist_noniid(train_dataset, args.num_users)    


    return client_train_data, test_dataset, dict_users_train, dict_users_test, server_train_data






















######################################## this is for cifar100 ########################################
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

########################### print exp parameters ##################################
def exp_details(args):
    print('\nExperimental details:')
    # print(f'    Model     : {args.model}')
    print(f'  Private Dataset   : {args.dataset}')
    print(f'  Public Dataset    : {args.public_dataset}')
    print(f'  Communication Rounds : {args.epochs}')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local training Epochs : {args.local_ep}\n')
    return