


# import torch
import torch
from torch import nn 
from torch.utils.data import DataLoader




# class Server(object):
#     def __init__(self, args, dataset):
#         self.args = args
#         self.device = args.device
#         self.train_loader = self.train_loader(dataset)
#         self.criterion = nn.CrossEntropyLoss().to(self.device)
    
#     def train_loader(self, dataset):

#         trainloader = DataLoader(dataset, batch_size=self.args.server_bs,
#                                           shuffle=True, num_workers=2)
        
#         return trainloader
    

#     def train_server(self, model, global_round):
#         model.to(self.device)
#         model.train()
#         epoch_loss = []
#         if self.args.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr,
#                         momentum = 0.5)
#         elif self.args.optimizer == 'adam':
#             optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr,
#                         weight_decay = 1e-4)
#         for iter in range(self.args.server_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 model.zero_grad()
#                 log_probs = model(images)
#                 loss = self.criterion(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 if self.args.verbose and ((batch_idx)%10 == 0):
#                     print('| Coommunication Rround: {} | Server Training Epoch: {}| [{}/{}]({:.0f}%)]\tLoss:{:.6f}'.format(
#                         global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
#                         100.* batch_idx/len(self.trainloader), loss.item()
#                     ))
#                 batch_loss.append(loss)
#             epoch_loss.append(sum(batch_loss)/len(batch_loss)) # each epoch loss is the avg of the batch loss
#         return model.state_dict(), sum(epoch_loss)/len(epoch_loss)
    



        

        