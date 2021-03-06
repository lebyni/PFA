from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch
import torchvision.datasets as datasets
import numpy as np


# myresnet=resnet50(pretrained=True)
# print (myresnet)


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16_bn(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = True

        self.avpool = nn.AvgPool2d(7, stride=1)
        classifier = nn.Sequential(
            nn.Linear(512, 1000),
        )
        self.fc = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


mean_list = np.zeros((1, 96))


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):

        global mean_list
        mean_temp = np.zeros((x.size(0), 1))
        # var_temp = np.zeros((x.size(0), 1))
        for name, module in self.submodule.features._modules.items():
            # if name is "fc": x = x.view(x.size(0), -1)
            if name in self.extracted_layers:

                x = module.conv[0](x)
                x = module.conv[1](x)
                x = module.conv[2](x)

                temp = x.cpu().detach().numpy()
                density = np.count_nonzero(temp, (2, 3))
                sparsity = (np.size(temp, 2) * np.size(temp, 3) - density) / (np.size(temp, 2) * np.size(temp, 3))
                mean_temp = np.concatenate((mean_temp, sparsity), axis=1)
                x = module.conv[3](x)
                x = module.conv[4](x)
                x = module.conv[5](x)
                temp = x.cpu().detach().numpy()
                density = np.count_nonzero(temp, (2, 3))
                sparsity = (np.size(temp, 2) * np.size(temp, 3) - density) / (np.size(temp, 2) * np.size(temp, 3))
                mean_temp = np.concatenate((mean_temp, sparsity), axis=1)

                x = module.conv[6](x)
                x = module.conv[7](x)


                # var_temp = np.concatenate((var_temp, var), axis=1)


            else:
                x = module(x)  # last layer output put into current layer input
        mean_temp = np.delete(mean_temp, 0, 1)
        # var_temp = np.delete(var_temp, 0, 1)

        mean_list = np.concatenate((mean_list, mean_temp), axis=0)
        # var_list = np.concatenate((var_list, var_temp), axis=0)




if __name__ == '__main__':

    path_dir = []
    # test_path = '/mnt/data/liuby/aaai-privacy-model-adapt/office-home/federated_learning/client'
    test_path = '/mnt/data/liuby/aaai-privacy-model-adapt/cifar10png/client'

    for i in [1,2,6,7,11,12,16,17,21,22]:
        tardir = test_path + str(i) + '/'
        path_dir.append(tardir)
    for client in range(10):
        mean_list = np.zeros((1, 96))
        var_list = np.zeros((1, 96))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path_dir[client], transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=64, shuffle=False, drop_last=False,
            num_workers=4)

        layer_number = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        # exact_list=['3','7','10']
        # exact_list = ['14', '17', '20']
        exact_list = ['2']
        # exact_list = ['34', '37', '40']
        # model=models.vgg16(pretrained=True).cuda()
        model = torch.load("/home/liuby/privacy-model-adapt/adapt_fl/mobilenet/cifar10/global_model/local5_ep50/localep50.pt",
                           map_location='cpu').cuda()
        model.eval()
        myexactor = FeatureExtractor(model, exact_list)

        output = []
        i = 0
        for data, target in test_loader:
            # print(data.size(0))
            data, target = data.cuda(), target.cuda()
            if i % 10 == 0:
                print(i)
            # t0 = time.time()
            myexactor(data)
            i += 1
            # breakv

        print(mean_list.shape)

        mean_list = np.delete(mean_list, 0, 0)

        print(mean_list.shape)

        # np.savetxt('./res18/office-home/feature_extract/result_relu/art_target_mean_layer3', mean_list)
        # np.savetxt('./res18/office-home/feature_extract/result_relu/art_target_var_layer3', var_list)
        file_name = './mobilenet/office_home/sparsity_extract/client' + str(client) + '_layer2_ratio_mobilenet'
        np.savetxt(file_name, mean_list)

