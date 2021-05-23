#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import pickle
import time
from torchvision import models, transforms

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import get_dataset, average_weights, exp_details
from mobilenetv2 import mobilenetv2
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ModifiedVGG11Model,MobileNet,ModifiedAlexnetModel
from options import args_parser

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    # device = 'cuda' if args.gpu else 'cpu'
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    # load dataset and user groups
    test_dir, user_dir = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'office-home':
            #cifar 32*32
            # global_model = ModifiedVGG11Model(args=args)

            # global_model = mobilenetv2(width_mult=0.25)
            # global_model.load_state_dict(torch.load('mobilenetv2_0.25-b61d2159.pth',map_location='cpu'))
            # global_model.classifier = torch.nn.Linear(in_features=1280, out_features=args.num_classes)

            # global_model=models.resnet18(pretrained=False)cd
            # global_model.fc = torch.nn.Linear(in_features=512, out_features=args.num_classes)
            global_model = torch.load('/home/liuby/privacy-model-adapt/adapt_fl/mobilenet/office_home/global_model/local5_ep50_0.25/localep50.pt',map_location='cpu')

    # elif args.model == 'mlp':
    #     # Multi-layer preceptron
    #     img_size = train_dataset[0][0].shape
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #         global_model = MLP(dim_in=len_in, dim_hidden=64,
    #                            dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    test_acc, test_loss = [], []
    best_acc=0
    best_epoch=0
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        idxs_users=args.num_users

        for idx in range(idxs_users):
            local_model = LocalUpdate(args=args, test_dir=test_dir,
                                      user_dir=user_dir[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            # loss=loss.cpu()
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        # global_weights=global_weights.cuda()
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, test_dir=test_dir,
                                      user_dir=user_dir[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            # acc, loss =acc.cpu(),loss.cpu()
            list_acc.append(acc)
            # list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print(f'Training Loss : {loss_avg}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


        # Test inference
        acc, loss = test_inference(args, global_model, test_dir)
        # acc, loss = acc.cpu(), loss.cpu()
        test_acc.append(acc)
        test_loss.append(loss)
        print('Test Accuracy: {:.2f}% \n'.format(100 * acc))
        print('test loss is ',loss)
        if acc>best_acc:
            best_epoch=epoch
            best_acc=acc
            best_model = copy.deepcopy(global_model)


    print('the best acc is',best_acc)
    print('the best epoch is ',best_epoch)
    # torch.save(best_model,'./mobilenet/office_home/random/real/localep5_ep50_50.pt')
    # np.savetxt('./mobilenet/office_home/random/real/localep5_ep50_train_loss_50',train_loss)
    # np.savetxt('./mobilenet/office_home/random/real/localep5_ep50_train_acc_50', train_accuracy)
    # np.savetxt('./mobilenet/office_home/random/real/localep5_ep50_test_loss_50', test_loss)
    # np.savetxt('./mobilenet/office_home/random/real/localep5_ep50_test_acc_50', test_acc)

    # torch.save(best_model, './mobilenet/cifar10_32/global_model/local5_ep50/localep50_50.pt')
    # np.savetxt('./mobilenet/cifar10_32/global_model/local5_ep50/localep50_train_loss_50', train_loss)
    # np.savetxt('./mobilenet/cifar10_32/global_model/local5_ep50/localep50_train_acc_50', train_accuracy)
    # np.savetxt('./mobilenet/cifar10_32/global_model/local5_ep50/localep50_test_loss_50', test_loss)
    # np.savetxt('./mobilenet/cifar10_32/global_model/local5_ep50/mobilenet_cifar_test', test_acc)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
