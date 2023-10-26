import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import os.path
from torch.utils.data import DataLoader, Dataset

def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)
    batch_loss = []

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss += batch_loss.item()
        batch_loss.append(loss.item())

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, sum(batch_loss)/len(batch_loss)

