from pyaad import Loda
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.io
import tqdm as tqdm
import numpy as np
import argparse
import psutil
import time
import os

process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
args = parser.parse_args()  

nfile = None
lfile = None
flag = 0

DATA_DIR = "../data/"
LABELS_DIR = "../data/labels/"

if args.dataset == 'NSL':
    nfile = DATA_DIR + 'data/memnslnumeric.txt'
    lfile = LABELS_DIR + 'memnsllabel.txt'
elif args.dataset == 'KDD': 
    nfile = DATA_DIR + 'data/memkddnumeric.txt'
    lfile = LABELS_DIR + 'memkddlabel.txt'
elif args.dataset == 'UNSW': 
    nfile = DATA_DIR + 'data/memunswnewnumeric.txt'
    lfile = LABELS_DIR + 'memunswlabel.txt'
elif args.dataset == 'DOS': 
    nfile = DATA_DIR + 'data/memdosnewnumeric.txt'
    lfile = LABELS_DIR + 'memdoslabel.txt'
elif args.dataset == 'IDS': 
    nfile = DATA_DIR + 'data/icsx_data.csv'
    lfile = LABELS_DIR + 'icsx_label.csv'
elif args.dataset == 'SYN':
    nfile = DATA_DIR + 'data/datanewnumeric.txt'
    lfile = LABELS_DIR + 'datalabel.txt'
else:
    nfile = DATA_DIR + 'New/' + args.dataset + '.mat'
    mat = scipy.io.loadmat(nfile)
    flag = 1


if flag == 0:
    X = np.loadtxt(nfile, delimiter = ',')
    labels = np.loadtxt(lfile, delimiter=',')
    if args.dataset == 'KDD':
        labels = 1 - labels

else:
    X = mat['X']
    labels = mat['y']

if len(X.shape) == 1:
    X = X.reshape(-1,1)

percentage = sum(labels)/len(labels)

print(args.dataset)
t = time.time()

BATCH_SIZE = min(1024, int(len(X)/10))

clf = Loda(random_state=0)

clf.fit(X[:BATCH_SIZE])
idx = BATCH_SIZE
y_preds = list(clf.decision_function(X[:BATCH_SIZE]))
batch = 0
total_batches = int(len(X)/BATCH_SIZE)

while idx + BATCH_SIZE < len(X):
    clf.fit(X[idx : idx + BATCH_SIZE])
    temp = clf.decision_function(X[idx : idx + BATCH_SIZE])
    y_preds += list(temp)
    # print("Batch = ", batch, "/", total_batches)
    batch += 1
    idx += BATCH_SIZE

clf.fit(X[idx: ])
temp = list(clf.decision_function(X[idx:]))
y_preds += temp

scores = y_preds
print("Time Taken", time.time() - t)        

np.savetxt("./scores/"+args.dataset+"_hst.csv",scores,delimiter=",")

auc = metrics.roc_auc_score(labels, scores)
count = int(np.sum(labels))
preds = np.zeros_like(labels)
indices = np.argsort(scores, axis=0)[::-1]
preds[indices[:count]] = 1
f1 = metrics.f1_score(labels, preds)
print("F1", f1, "AUC", max(auc, 1-auc))
print("Confusion Matrix", metrics.confusion_matrix(labels, preds))
something = (1 - labels)*scores
something = something[np.nonzero(1-labels)]
normal = np.sort(something)
something = labels*scores
something = something[np.nonzero(labels)]
anomaly = np.sort(something)
print("Normal stats", np.median(normal), np.max(normal), np.min(normal), np.mean(normal))
print("Anomaly stats", np.median(anomaly), np.max(anomaly), np.min(anomaly), np.mean(anomaly))
print("RSS memory used = ", process.memory_info().rss/2**20)
