#!/bin/bash



sudo python3 federated_main.py --model=cnn --dataset=office-home --gpu=1 --num_classes=65 --local_ep=5 --local_bs=32 --num_users=20 --epochs=30 --lr=0.01



