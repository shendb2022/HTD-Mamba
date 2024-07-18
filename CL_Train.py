import os
import time
from typing import Dict
import torch
import numpy as np
from ts_generation import ts_generation
from Data_Patch import Data
import matplotlib.pyplot as plt
from CL_Model import SpectralGroupAttention
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Utils import checkFile, standard
from Scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import scipy.io as sio
from sklearn import metrics
import random


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('loss-trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()


def train(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    dataset = Data(modelConfig["path"], w_size=modelConfig['patch_size'])
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=True)
    # model setup
    net_model = SpectralGroupAttention(band=modelConfig['band'], group_length=modelConfig['m'],
                                       channel_dim=modelConfig['channel'], state_size=modelConfig['state_size'],
                                       device=device, layer=modelConfig['layer']).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    path = modelConfig["save_dir"] + '/' + modelConfig['dataset'] + '/'
    checkFile(path)
    # start training
    net_model.train()
    loss_list = []
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for center_pixels, coded_vectors in tqdmDataLoader:
                # train
                combined_vectors = torch.cat([center_pixels, coded_vectors], dim=0)
                optimizer.zero_grad()
                x_0 = combined_vectors.to(device)
                # print(x_0.shape)
                features = net_model(x_0)
                loss = info_nce_loss(features, modelConfig['batch_size'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e % 1 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                path, 'ckpt_' + str(e) + "_.pt"))
        loss_list.append(loss.item())
    paintTrend(loss_list, epochs=modelConfig['epoch'], stride=20)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def cosin_similarity(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def transpose(x):
    return x.transpose(-2, -1)


def info_nce_loss(x, size, temperature=0.1, reduction='mean'):
    batch_x0 = x[:size]
    batch_x1 = x[size:]
    batch_x0, batch_x1 = normalize(batch_x0, batch_x1)
    logits = batch_x0 @ transpose(batch_x1)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(batch_x0), device=batch_x0.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def select_best(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    opt_epoch = 0
    max_auc = 0
    path = modelConfig["save_dir"] + '/' + modelConfig['dataset'] + '/'
    for e in range(0, modelConfig['epoch'], 1):
        with torch.no_grad():
            mat = sio.loadmat(modelConfig["path"])
            data = mat['data']
            map = mat['map']
            data = standard(data)
            target_spectrum = ts_generation(data, map, 7)
            h, w, c = data.shape
            numpixel = h * w
            data_matrix = np.reshape(data, [-1, c], order='F')
            model = SpectralGroupAttention(band=modelConfig['band'], group_length=modelConfig['m'],
                                           channel_dim=modelConfig['channel'], state_size=modelConfig['state_size'],
                                           device=device, layer=modelConfig['layer'])
            model = model.to(device)
            ckpt = torch.load(os.path.join(
                path, "ckpt_%s_.pt" % e), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.%s" % e)
            model.eval()

            batch_size = modelConfig['batch_size']
            detection_map = np.zeros([numpixel])
            target_prior = torch.from_numpy(target_spectrum.T)
            target_prior = target_prior.to(device)
            target_prior = torch.unsqueeze(target_prior, dim=1)
            target_features = model(target_prior)
            target_features = target_features.cpu().detach().numpy()

            for i in range(0, numpixel - batch_size, batch_size):
                pixels = data_matrix[i:i + batch_size]
                pixels = torch.from_numpy(pixels)
                pixels = pixels.to(device)
                pixels = torch.unsqueeze(pixels, dim=1)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[i:i + batch_size] = cosin_similarity(features, target_features)

            left_num = numpixel % batch_size
            if left_num != 0:
                pixels = data_matrix[-left_num:]
                pixels = torch.from_numpy(pixels)
                pixels = pixels.to(device)
                pixels = torch.unsqueeze(pixels, dim=1)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[-left_num:] = cosin_similarity(features, target_features)

            detection_map = np.reshape(detection_map, [h, w], order='F')
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)
            y_l = np.reshape(map, [-1, 1], order='F')
            y_p = np.reshape(detection_map, [-1, 1], order='F')

            ## calculate the AUC value
            try:
                fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
                fpr = fpr[1:]
                tpr = tpr[1:]
                auc = round(metrics.auc(fpr, tpr), modelConfig['epision'])
            except:
                auc = 0.5
            if auc > max_auc:
                max_auc = auc
                opt_epoch = e
    print(max_auc)
    print(opt_epoch)
    return max_auc

def eval(modelConfig: Dict):
    start = time.perf_counter()
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    path = modelConfig["save_dir"] + '/' + modelConfig['dataset'] + '/'
    with torch.no_grad():
        mat = sio.loadmat(modelConfig["path"])
        data = mat['data']
        map = mat['map']
        data = standard(data)
        target_spectrum = ts_generation(data, map, 7)
        h, w, c = data.shape
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')
        model = SpectralGroupAttention(band=modelConfig['band'], group_length=modelConfig['m'],
                                       channel_dim=modelConfig['channel'], state_size=modelConfig['state_size'],
                                       device=device, layer=modelConfig['layer'])
        model = model.to(device)
        ckpt = torch.load(os.path.join(
            path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        batch_size = modelConfig['batch_size']
        detection_map = np.zeros([numpixel])
        target_prior = torch.from_numpy(target_spectrum.T)
        target_prior = target_prior.to(device)
        target_prior = torch.unsqueeze(target_prior, dim=1)
        target_features = model(target_prior)
        target_features = target_features.cpu().detach().numpy()

        for i in range(0, numpixel - batch_size, batch_size):
            pixels = data_matrix[i:i + batch_size]
            pixels = torch.from_numpy(pixels)
            pixels = pixels.to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity(features, target_features)

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = data_matrix[-left_num:]
            pixels = torch.from_numpy(pixels)
            pixels = pixels.to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity(features, target_features)

        detection_map = np.exp(-1 * (detection_map - 1) ** 2 / modelConfig['delta'])
        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)
        end = time.perf_counter()
        print('excuting time is %s' % (end - start))
        # save_path = '/home/sdb/experiments/20240504/%s/HTD-Mamba.mat' % modelConfig['dataset']
        # sio.savemat(save_path, {'map': detection_map})
        y_l = np.reshape(map, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        ## calculate the AUC value
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]
        auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
        auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
        auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
        auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
        auc5 = round(auc3 / auc2, modelConfig['epision'])
        print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))

        plt.imshow(detection_map)
        plt.show()
