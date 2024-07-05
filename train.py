# coding:utf-8
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
import argparse
from utils import ECCLoss
import config
import time
import warnings

warnings.filterwarnings("ignore")


def train_one_epoch(model, loss_function, epoch, train_loader, device, optimizers, conf):
    train_len = len(train_loader.dataset)
    all_train_step = int(train_len / conf.batch_size)

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    ratio = epoch / conf.epochs

    print(datetime.datetime.now())
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        logits, feature = model(inputs)

        if conf.loss == 'ecc':
            loss1 = loss_function[0](logits, labels)
            feature_center_loss, logit_center_loss, feature_table, logits_table = loss_function[1](feature, logits, labels)
            loss = loss1 + conf.lmd_1 * ratio * feature_center_loss + conf.lmd_2 * ratio * logit_center_loss

        elif conf.loss == 'celoss':
            loss = loss_function(logits, labels)
        else:
            raise ValueError("no this loss")

        _, predict = torch.max(logits, dim=1)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        running_loss += loss.item()
        running_acc += torch.sum(predict == labels.data).item()

        if step % conf.step_distance == 0:
            print("{}/{} train loss: {:.3f} train acc: {:.3f} || lr: {:.6f} ".format(step,
                                                                                     all_train_step,
                                                                                     running_loss / train_len,
                                                                                     running_acc / train_len,
                                                                                     optimizers[0].state_dict()[
                                                                                         'param_groups'][0]['lr'],

            ))
    print("train loss: {:.3f} train acc: {:.3f}".format(
        running_loss / train_len,
        running_acc / train_len))


def val(model, loss_function, val_loader, device, conf, epoch):

    val_len = len(val_loader.dataset)

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        ratio = epoch / conf.epochs
        for step, data in enumerate(val_loader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, feature = model(inputs)

            if conf.loss == 'ecc':
                loss = loss_function[0](logits, labels)
            elif conf.loss == 'celoss':
                loss = loss_function(logits, labels)
            else:
                raise ValueError("no this loss")
            _, predict = torch.max(logits, dim=1)
            running_loss += loss.item()
            running_acc += torch.sum(predict == labels.data).item()

        print("val loss: {:.3f} val acc: {:.3f}".format(running_loss / val_len, running_acc / val_len))
        return running_acc / val_len


def main(argument):
    conf = config.Config(argument)
    conf.print_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf.loss == 'ecc':
        loss_function = [nn.CrossEntropyLoss().to(device),
                         ECCLoss(conf.num_class, conf.dim).to(device)]
    elif conf.loss == 'celoss':
        loss_function = nn.CrossEntropyLoss().to(device)
    else:
        raise NotImplementedError('no this loss function!')

    train_loader, val_loader = conf.train_dataloader, conf.val_dataloader
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    print(f"train image num: {train_len} val image num: {val_len}")

    model = conf.model
    if 'densenet' in conf.model_name:
        parts_params = list(model.classifier.parameters())
        parts_params_id = list(map(id, model.classifier.parameters()))
    elif 'convnext' in conf.model_name:
        parts_params = list(model.head.parameters())
        parts_params_id = list(map(id, model.head.parameters()))
    else:
        parts_params = list(model.fc.parameters())
        parts_params_id = list(map(id, model.fc.parameters()))

    base_params = filter(lambda p: id(p) not in parts_params_id, model.parameters())
    if conf.dataset_name == 'cubbirds':
        params = [
            {'params': parts_params, 'lr': conf.lr},
            {'params': base_params, 'lr': 0.1 * conf.lr}
        ]
    else:
        params = model.parameters()

    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = opt.SGD(params, lr=conf.lr, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)

    optimizers = [optimizer]

    print('start training')
    max_acc = 0.0
    max_epoch = 0
    for epoch in range(conf.epochs):
        print('Epoch {}/{}'.format(epoch + 1, conf.epochs))
        print('-' * 10)
        train_one_epoch(model, loss_function, epoch, train_loader, device, optimizers, conf)
        acc = val(model, loss_function, val_loader, device, conf, epoch)

        if (acc > max_acc) and (conf.model_path != 'no_save'):
            hyperparam = f'_{conf.img_size}_{conf.lr}_{conf.lmd_1}_{conf.lmd_2}_{conf.loss}_{conf.text}_{conf.dataset_name}_{conf.time}'
            model_save_path = conf.model_path + '/' + conf.model_name + hyperparam + '_{:.4f}.pth'.format(max_acc)

            if os.path.exists(model_save_path):
                os.remove(model_save_path)

            max_acc = acc
            max_epoch = epoch
            torch.save(model.state_dict(), conf.model_path + '/' + conf.model_name + hyperparam + '_{:.4f}.pth'.format(max_acc))
        scheduler.step()

    return max_epoch, max_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=448, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--psd', default=100, type=int)

    parser.add_argument('--lmd_0', default=0.0, type=float)
    parser.add_argument('--lmd_1', default=0.0, type=float)
    parser.add_argument('--lmd_2', default=0.0, type=float)
    parser.add_argument('--lmd_3', default=0.0, type=float)
    parser.add_argument('--lmd_4', default=0.0, type=float)
    parser.add_argument('--lmd_5', default=0.0, type=float)
    parser.add_argument('--k', default=1, type=int)

    parser.add_argument('--dataset_name', default='aircrafts',type=str)
    parser.add_argument('--model_path', default='centerloss', type=str)
    parser.add_argument('--loss', default='ecc', type=str)
    parser.add_argument('--text', default='bs32', type=str)
    parser.add_argument('--time', default=1, type=int)
    arg = parser.parse_args()

    for arg.time in range(3):
        max_epoch, max_acc = main(arg)
        all_epoch = []
        all_acc = []

