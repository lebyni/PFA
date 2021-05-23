#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torchvision.datasets as datasets
# from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from copy import deepcopy

criterion = torch.nn.CrossEntropyLoss()




class LocalUpdate(object):
    def __init__(self, args, test_dir, user_dir, logger):
        self.args = args
        self.logger = logger
        self.trainloader,  self.testloader = self.train_val_test(test_dir,user_dir)

        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_val_test(self,test_dir, user_dir):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(user_dir, transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(32, padding=4),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(test_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.args.local_bs, shuffle=True,
            num_workers=4, pin_memory=True)

        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round, iter, batch_idx * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item()/self.args.local_bs)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def criterion_kd(self,outputs, targets, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = 0.95
        T = 6
        KD_loss = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(outputs / T, dim=1),
                                       torch.nn.functional.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
                  torch.nn.functional.cross_entropy(outputs, targets) * (1. - alpha)
        return KD_loss

    def update_weights_kd(self, model,teacher_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                teacher_model=teacher_model.cuda()
                log_probs = model(images)
                teacher_outputs = teacher_model(images)
                loss = self.criterion_kd(log_probs, labels,teacher_outputs)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round, iter, batch_idx * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item()/self.args.local_bs)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def criterion_ewc(self,global_model, model, fisher, output, targets, criterion, lamb=5000):
        model_old = deepcopy(global_model).to(self.device)
        model_old.eval()
        # model.cpu()
        for param in model_old.parameters():  # Freeze the weights
            param.requires_grad = False
        # Regularization for all previous tasks
        loss_reg = 0
        for (name, param), (_, param_old) in zip(model.named_parameters(), model_old.named_parameters()):
            loss_reg += torch.sum(fisher[name].to(self.device) * (param_old - param).pow(2)) / 2
        # model.cuda()
        # loss_reg.cuda()
        model_old.cpu()
        # print(type(loss_reg))
        return criterion(output, targets) + lamb * loss_reg


    def update_weights_ewc(self, model,teacher_model,fisher, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                # teacher_model = teacher_model.cuda()
                loss = self.criterion_ewc(teacher_model, model, fisher, log_probs, labels, criterion)
                # loss.to(self.device)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round, iter, batch_idx * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item()/self.args.local_bs)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        loss=loss/len(self.trainloader.dataset)
        return accuracy, loss






def test_inference(args, model, test_dir):
    """ Returns the test accuracy and loss.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(size_average=False)
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss / len(testloader.dataset)
    return accuracy, loss
