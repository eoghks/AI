#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import xmltodict
import base64
import numpy as np
import array

from tqdm import tqdm

from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
from IPython.display import clear_output

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', dest='path', help='dataset path', default='./')
    parser.add_argument('--gpu', dest='gpu', help='the number of gpu to use', default=0, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate value', default=5e-6, type=float)
    parser.add_argument('--dropout', dest='dropout', help='drop out', default=0.3, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=32, type=int)
    parser.add_argument('--training_epoch', dest='training_epoch', help='training_epoch', default=100, type=int)
    parser.add_argument('--test_model_weights', dest='test_model_weights', help='dataset path', default='./best_cnn_model.h5')
    parser.add_argument("--test", dest='test',action="store_true", help="Use model test")

    args = parser.parse_args()
    return args

# Hyperparameter
# 학습에 필요한 하이퍼파라미터들
args = parse_args()
path = args.path
gpu = args.gpu
lr = args.lr
drop_out = args.dropout
batch_size = args.batch_size
training_epoch = args.training_epoch
device = torch.device(f'cuda:{str(gpu)}' if torch.cuda.is_available() else 'cpu')
epoches = training_epoch
SEPARATOR = '======================================='

print(SEPARATOR)
print("Hyperparameter")
print(f"path              : {path}")
print(f"gpu               : {gpu}")
print(f"learning rate     : {lr}")
print(f"drop_out          : {drop_out}")
print(f"batch_size        : {batch_size}")
print(f"training_epoch    : {training_epoch}")
print(f"test              : {args.test}")
if args.test:
    print(f"test_model_weight : {args.test_model_weights}")
print(SEPARATOR)

# Data Preprocessings
def get_lead(path):
    with open(path, 'rb') as xml:
        ECG = xmltodict.parse(xml.read().decode('utf8'))
    
    augmentLeads = True
    if path.split('/')[-1][0] == '5':
        waveforms = ECG['RestingECG']['Waveform'][1]
    elif path.split('/')[-1][0] == '6':
        waveforms = ECG['RestingECG']['Waveform']
        augmentLeads = False
    else:
        waveforms = ECG['RestingECG']['Waveform']
    
    leads = {}
    
    for lead in waveforms['LeadData']:
        lead_data = lead['WaveFormData']
        lead_b64  = base64.b64decode(lead_data)
        lead_vals = np.array(array.array('h', lead_b64))
        leads[ lead['LeadID'] ] = lead_vals
    
    if augmentLeads:
        leads['III'] = np.subtract(leads['II'], leads['I'])
        leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
        leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
        leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
    
    return leads

# 에러 파일들
error_files = ['6_2_003469_ecg.xml', '6_2_003618_ecg.xml', '6_2_005055_ecg.xml', '8_2_001879_ecg.xml', '8_2_002164_ecg.xml', '8_2_007281_ecg.xml', '8_2_008783_ecg.xml', '8_2_007226_ecg.xml']

# 데이터
train_data = []
train_labels = []
valid_data = []
valid_labels = []


# 데이터셋 경로들
train_pathes = [path+'data/train/arrhythmia/', path+'data/train/normal/']
valid_pathes = [path+'data/validation/arrhythmia/', path+'data/validation/normal/']

error_decode = []   # 디코딩에 실패한 데이터들..

print(SEPARATOR)
print("Data preprocessing Start!")
print(SEPARATOR)
# 데이터셋 구성
for idx1, pathes in enumerate([train_pathes, valid_pathes]):
    
    for path in pathes:
        for file in os.listdir(path):

            if file in error_files or 'ipynb' in file:
                continue

            try:
                data = get_lead(path + file)
            except Exception as e:
                error_decode.append(path + file)

            listed_data = []
            keys = sorted(data.keys())
            for key in keys:
                listed_data.append(data[key])

            for idx2, i in enumerate(listed_data):
                if len(i) != 5000:
                    listed_data[idx2] = np.append(i, np.zeros(5000-len(i)))

                    
            # save each train, valid data
            if idx1== 0: 
                train_data.append(listed_data)
                if 'arrhythmia' in path:
                    train_labels.append(1)
                else:
                    train_labels.append(0)
            else:
                valid_data.append(listed_data)
                if 'arrhythmia' in path:
                    valid_labels.append(1)
                else:
                    valid_labels.append(0)

# lead 개수가 12개가 아닌 데이터 찾기
error_lead_len = []
for idx, i in enumerate(train_data):
    if len(i) != 12:
        error_lead_len.append(idx)
for i in error_lead_len:
    del train_data[i]
    del train_labels[i]



# # 데이터 길이 및 lead 개수 분석

#  데이터의 길이 분포 확인: valid는 모두 5000인것을 확인
#  
#  train은 60912개가 4999, 36개가 1249, 1개가 4988
#  
#  위의 테스크는 먼저 4999개만 0의 패딩을 붙이고 나머지는 제외하는식으로 전처리함
# 딱 한개의 9 lead의 데이터가 존재한다..

# # Dataset 생성
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels))
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_data).float(), torch.tensor(valid_labels))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)


class Classifier(nn.Module):
    def __init__(self, drop_out=0.0):
        super(Classifier, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        self.pool1 = nn.MaxPool1d(4)
        self.pool2 = nn.MaxPool1d(5)
        self.pool3 = nn.MaxPool1d(5)

        self.fc1 = nn.Linear(128 * 50, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.relu(self.cnn1(x))
        x = self.pool1(x)
        x = self.relu(self.cnn2(x))
        x = self.pool2(x)
        x = self.relu(self.cnn3(x))
        x = self.pool3(x)

        x = x.view(-1, 128 * 50)

        x = self.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.relu(self.fc3(x))
        x = self.drop_out(x)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        x = torch.sigmoid(x)

        return x.view(-1)


model = Classifier(drop_out=drop_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.BCELoss()

# best model을 저장하기 위한 변수들
best_auc = 0
best_epoch = -1
best_pred = []

prev_model = None

if not args.test:
    print(SEPARATOR)
    print("Training Start!")
    for i in tqdm(range(epoches)):

        # Train
        loss_sum = 0
        true_labels = []
        pred_labels = []
        model.train()
        for e_num, (x,y) in enumerate(train_dataloader):
            x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
            model.zero_grad()
            pred_y = model(x)

            loss=criterion(pred_y,y)
            loss_sum+=loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

        auc = roc_auc_score(true_labels,pred_labels)

        # Valid
        loss_sum=0
        true_labels=[]
        pred_labels=[]
        model.eval()
        for e_num, (x,y) in enumerate(val_dataloader):
            x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

            pred_y = model(x)
            loss=criterion(pred_y,y)

            loss_sum+=loss.detach()

            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

        auc = round(roc_auc_score(true_labels,pred_labels), 6)

        if auc > best_auc:
            best_pred = pred_labels
            best_auc = auc
            best_epoch = i

            if prev_model is not None:
                os.remove(prev_model)
            prev_model = f'cnn_model_{best_auc}.h5'
            torch.save(model.state_dict(), prev_model)

    print(f'best validation acc = {best_auc}, in epoch {best_epoch}')

else:
    print(SEPARATOR)
    print("Test Start!")
    model.load_state_dict(torch.load(args.test_model_weights))
    model.eval()

    loss_sum = 0
    true_labels = []
    pred_labels = []
    model.eval()

    for e_num, (x, y) in enumerate(val_dataloader):
        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

        pred_y = model(x)
        loss = criterion(pred_y, y)

        loss_sum += loss.detach()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    best_pred = pred_labels
    auc = roc_auc_score(true_labels, pred_labels)
    print("AUC :", auc)

# 결과 plot
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='blue', label='ROC')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()


import  matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fper, tper, thresholds = roc_curve(true_labels,best_pred)
plot_roc_curve(fper, tper)