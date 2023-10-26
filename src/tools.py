import torch
import torch.nn as nn
import torch.nn.functional as F

def splitList (lst, n):
    it = iter(lst)
    new = [[next(it) for _ in range(n)] for _ in range(len(lst) // n)]

    for i, x in enumerate(it):
        new[i].append(x)

    return new

def model_generation(client_number):

    model_short_name = ['CNN1', 'CNN2', 'CNN3', 'CNN4']
    
    client_index_list = [i for i in range(client_number)]
    new_list = splitList(client_index_list, 3)
    client_model_dict = {}
    for j in range (len(new_list)):
        for item in new_list[j]:
            client_model_dict[item] = model_short_name[j]
    return client_model_dict # key is the client index, value is 'a''b'...


def model_assign(client_number, client_id):
    if model_generation(client_number)[client_id] == 'CNN1':
        client_model = CNN1()
    if model_generation(client_number)[client_id] == 'CNN2':
        client_model = CNN2()
    if model_generation(client_number)[client_id] == 'CNN3':
        client_model = CNN3()
    if model_generation(client_number)[client_id] == 'CNN4':
        client_model = CNN4()
    # if model_generation(client_number)[client_id] == 'CNN1':
    #     client_model = CNN5()
    return client_model


class Combined_model(nn.Module):
    def __init__(self, layer_list, size_list, input_size, num_class=10):
        super(Combined_model, self).__init__()
        self.size_list = size_list
        self.layers = nn.ModuleList()
        for lay in layer_list:
            self.layers.append(lay)

        self.mlps = nn.ModuleList()
        for i in range(len(size_list)):
            if i == 0:
                mlp_out_size = size_list[i][0][-3]*size_list[i][0][-2]*size_list[i][0][-1]
                self.mlps.append(nn.Linear(input_size, mlp_out_size))
            else:
                mlp_in_size = 1
                for j in range(len(size_list[i-1][1])):
                    mlp_in_size = mlp_in_size * size_list[i-1][1][j]
                mlp_out_size = 1
                for j in range(1, len(size_list[i][0])):
                    mlp_out_size = mlp_out_size * size_list[i][0][j]
                self.mlps.append(nn.Linear(mlp_in_size, mlp_out_size))
        final_out_size = 1
        for k in range(len(size_list[-1][1])):
            final_out_size = final_out_size*size_list[-1][1][k]
        self.classifier = nn.Linear(final_out_size, num_class)

    def forward(self, x):
        bs = x.size(0)
        for i in range(len(self.mlps)):
            x = x.view(bs, -1)
            x = self.mlps[i](x)
            sizes = [size for size in self.size_list[i][0]][1:]
            shape_tuple = [bs] + [s for s in sizes]
            x = x.view(shape_tuple)
            x = self.layers[i](x)
        return self.classifier(x)


class CNN1_1(nn.Module):
    def __init__(self, args):
        super(CNN1_1, self).__init__()
        # self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
class CNN2_1(nn.Module):
    def __init__(self, args):
        super(CNN2_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*2*2, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
# model1 =  CNN1()
# model2 =  CNN2()
# # model3 =  CNN3()

# dict_model1 = {0: model1.conv1(), 1: model1.conv2(), 2: model1.fc1(), 3: model1.fc2()}
# dict_model2 = {0: model2.conv1(), 1: model2.conv2(), 2: model2.fc1(), 3: model2.fc2()}
# # dict_model3 = {0: model3.conv1(), 1: model3.conv2(), 2: model3.fc1(), 3: model3.fc2()}

# dict_shape1 = {0: model1.conv1().shape, 1: model1.conv2().shape, 2: model1.fc1().shape, 3: model1.fc2().shape}

# class Combine(nn.Module):
#     def __init__(self, dict_model1, dict_shape1, dict_model2, index1, index2):
#         super().__init__()
#         self.layer1 = dict_model1[index1]
#         self.layer2 = dict_model2[index2]
        
#         self.mlp = nn.Linear(dict_shape1[index1], 10)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.mlp(x)
#         x = self.layer2(x)
        
#         return x

class CNN3_1(nn.Module):
    def __init__(self, args):
        super(CNN3_1, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU()
        )
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # self.conv4 = nn.Conv2d(32, 64, 5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU()
        )
        
        nn.Conv2d(64, 128, 5)
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        # self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.pool(x)
        x = self.conv5(x)
        # x = self.pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class CNN4_1(nn.Module):
    def __init__(self, args):
        super(CNN4_1, self).__init__()
        self.num_classes = args.num_classes
        self.output_dim = 512
        # Define a series of convolutional layers named conv1, conv2, etc.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.05)
        )

        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout2d(p=0.05)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        # self.fc_layer = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True)
        # )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(256, args.num_classes)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        # Pass the input tensor through the series of convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        # x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.pool2(x)
        # x = self.dropout1(x)
        # x = self.conv5(x)
        # x = self.bn3(x)
        # x = self.relu5(x)
        # x = self.conv6(x)
        # x = self.relu6(x)
        # x = self.pool3(x)
        # Flatten the output tensor and pass it through the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # Apply log softmax activation to the output
        return x


# class CNN4(nn.Module):
#     def __init__(self, args):
#         super(CNN4, self).__init__()
#         self.num_classes = args.num_classes
#         self.output_dim=512
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True)

#         )
#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layer(x)
#         return F.log_softmax(x, dim=1)



class CNN5(nn.Module):
    def __init__(self, args):
        super(CNN5, self).__init__()
        self.num_classes = args.num_classes
        self.output_dim=512
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)

        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)

















class CNN5_(torch.nn.Module): #VGG16

    def __init__(self, args):
        super(CNN5_, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          # (1(32-1)- 32 + 3)/2 = 1
                          padding=1), 
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),    
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, args.num_classes),
        )
            
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        
    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas

class CNN1(nn.Module):
    def __init__(self, args):
        super(CNN1, self).__init__()
        # self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
class CNN2(nn.Module):
    def __init__(self, args):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
# model1 =  CNN1()
# model2 =  CNN2()
# # model3 =  CNN3()

# dict_model1 = {0: model1.conv1(), 1: model1.conv2(), 2: model1.fc1(), 3: model1.fc2()}
# dict_model2 = {0: model2.conv1(), 1: model2.conv2(), 2: model2.fc1(), 3: model2.fc2()}
# # dict_model3 = {0: model3.conv1(), 1: model3.conv2(), 2: model3.fc1(), 3: model3.fc2()}

# dict_shape1 = {0: model1.conv1().shape, 1: model1.conv2().shape, 2: model1.fc1().shape, 3: model1.fc2().shape}

# class Combine(nn.Module):
#     def __init__(self, dict_model1, dict_shape1, dict_model2, index1, index2):
#         super().__init__()
#         self.layer1 = dict_model1[index1]
#         self.layer2 = dict_model2[index2]
        
#         self.mlp = nn.Linear(dict_shape1[index1], 10)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.mlp(x)
#         x = self.layer2(x)
        
#         return x

class CNN3(nn.Module):
    def __init__(self, args):
        super(CNN3, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU()
        )
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.conv4 = nn.Conv2d(32, 64, 5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU()
        )
        
        nn.Conv2d(64, 128, 5)
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        # self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, args.num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.pool(x)
        x = self.conv5(x)
        # x = self.pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class CNN4(nn.Module):
    def __init__(self, args):
        super(CNN4, self).__init__()
        self.num_classes = args.num_classes
        self.output_dim = 512
        # Define a series of convolutional layers named conv1, conv2, etc.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )

        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout2d(p=0.05)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        # self.fc_layer = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True)
        # )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.linear = nn.Linear(512, args.num_classes)
        # self.fc2 = nn.Linear(4096, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        # Pass the input tensor through the series of convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        # x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.pool2(x)
        # x = self.dropout1(x)
        # x = self.conv5(x)
        # x = self.bn3(x)
        # x = self.relu5(x)
        # x = self.conv6(x)
        # x = self.relu6(x)
        # x = self.pool3(x)
        # Flatten the output tensor and pass it through the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # Apply log softmax activation to the output
        return self.linear(x)
