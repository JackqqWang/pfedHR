import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--server_ep', type=int, default=50,help="number of rounds of training at the server side")
    parser.add_argument('--model_same', type=int, default=0,
                        help='1 is all the clients share the same model')
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of communication rounds")
    parser.add_argument('--sfine', type=int, default=3,
                        help="server fine tune")
    parser.add_argument('--num_users', type=int, default=12,
                        help="number of users: K")
    parser.add_argument('--expected_num_models', type=int, default=4,
                        help="number of constructed models")    
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='weight of the teacher and student')
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--sv_batch_size', type=int, default=5000,
                        help="server batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--customize_test', type=int, default=1,
                        help='Default set to customized. Set to 0 for share test data.')
    parser.add_argument('--model', type=str, default='CNN1', help='model name')

    parser.add_argument('--num_shards', type=int, default=200,
                        help='number of shards')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")

    # other arguments
    parser.add_argument('--metric_hyper', type=float, default=0.5, help="metric hyper")
    parser.add_argument('--cluster_num', type=int, default=2, help="number \
                        of cluster classes")

    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--public_dataset', type=str, default='cifar10', help="name \
                        of public dataset")
    parser.add_argument('--input_size', type=int, default=10, help="cifar10: 3*32*32 = 3072, svhn: 3072, mnist: 7841")   
    parser.add_argument('--supervised', default=1, help="1 is supervised, 0 is unsueprvised")

    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=1, help="To use cuda, set \
                        to 1. Default set  0 to use CPU.")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--gpu_id', default=0, help=" 0, 1, 2, 3")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
