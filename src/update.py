from unittest import TestLoader
import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset












class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image).clone().detach(), torch.tensor(label).clone().detach()



class LocalUpdate(object):

    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs)
        )
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def train_test(self, dataset, idxs):
        """
        return train and test dataloaders for given dataset and user indexes
        """
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size = self.args.local_bs, shuffle= True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size = self.args.local_bs, shuffle= True)

        return trainloader, testloader
    def k_distll(self,student_model,teacher_model):
        student_model.train()
        teacher_model.eval()
        student_model.to(self.device)
        teacher_model.to(self.device)
        epoch_loss = []
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(student_model.parameters(), lr = self.args.lr,
                        momentum = 0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(student_model.parameters(), lr = self.args.lr,
                        weight_decay = 1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                student_model.zero_grad()
                teacher_model.zero_grad()
                student_log_probs = student_model(images)
                teacher_log_probs = teacher_model(images)
                loss = self.criterion(student_log_probs, labels) + self.args.alpha * self.criterion(student_log_probs, teacher_log_probs)
                loss.backward()
                optimizer.step()
                if self.args.verbose and ((batch_idx)%10 == 0):
                    print('| KD Local Epoch: {}| [{}/{}]({:.0f}%)]\tLoss:{:.6f}'.format(
                        iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100.* batch_idx/len(self.trainloader), loss.item()
                    ))
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return student_model.state_dict(), sum(epoch_loss)/len(epoch_loss)



    def update_weights(self, model, global_round):
        model.to(self.device)
        model.train()
        epoch_loss = []
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr,
                        momentum = 0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr,
                        weight_decay = 1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and ((batch_idx)%10 == 0):
                    print('| Global Round: {} | Local Epoch: {}| [{}/{}]({:.0f}%)]\tLoss:{:.6f}'.format(
                        global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100.* batch_idx/len(self.trainloader), loss.item()
                    ))
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) # each epoch loss is the avg of the batch loss
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)





        