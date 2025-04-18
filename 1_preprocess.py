# -*- coding: utf-8 -*-
import os
import json
from numpy import ones, zeros, exp, argsort, flipud, dot, logical_and
from scipy.sparse import spdiags, eye, dia_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# The fluorescence background interference in the Raman spectrum was removed by invoking the background subtraction algorithm.
def WhittakerSmooth(x, lamb, w):
    m = w.shape[0]
    W = spdiags(w, 0, m, m)
    D = eye(m - 1, m, 1) - eye(m - 1, m)
    wh = spsolve((W + lamb * D.transpose() * D), w * x)
    return wh


# fitted background
def airPLS(x, lamb=100, itermax=10):
    m = x.shape[0]
    w = ones(m)
    for i in range(itermax):
        z = WhittakerSmooth(x, lamb, w)
        d = x - z
        if sum(abs(d[d < 0])) < 0.001 * sum(abs(x)):
            break;
        w[d < 0] = exp(i * d[d < 0] / sum(d[d < 0]))
        # w[d<0]=exp(i*d[d<0]/sum(abs(d[d<0]))
        w[d >= 0] = 0
    return z

# The background-subtracted spectrum was obtained by subtracting the fitted background from the original spectrum.
def airPLS_MAT(X, lamb=100, itermax=10):
    B = X.copy()
    for i in range(X.shape[0]):
        B[i,] = airPLS(X[i,], lamb, itermax)
    return X - B

def readInfo(path):
    name = os.listdir(path)
    return name

def locate_line(lines, keyword):  # find the line with keyword and return the index

    for i in range(0, len(lines)):
        fr = lines[i].find(keyword)
        if fr != -1:
            return i
    return "CAN'T FIND"


def get_content(path, filename):
    content = open(path + "/" + filename).read()
    filecontent = content.split('\n')
    title = locate_line(filecontent, 'Raman Shift')
    xunit = filecontent[title].split(';')
    AXI = xunit.index('Raman Shift')  # index of Raman Shift
    DSI = xunit.index('Dark Subtracted #1')  # index of Dark Subtracted #1
    xaxis_min = title + 1 + int(filecontent[locate_line(filecontent, 'xaxis_min')].split(';')[1])  # min and max of x axis
    xaxis_max = title + 1 + int(filecontent[locate_line(filecontent, 'xaxis_max')].split(';')[1])
    # nLen = xaxis_max - xaxis_min
    ds = []

    data = []
    for i in range(xaxis_min, xaxis_max):
        datacon = filecontent[i].split(';')
        rs = float(datacon[AXI])
        if 1200 < rs < 1400:
            ds.append(float(datacon[DSI]))
            data_s = []
            data_s.append(rs)
            data_s.append(float(datacon[DSI]))
            data.append(data_s)
    # data = np.array(data)
    # data = airPLS_MAT(data)
    # plt.plot(data[0], data[1])
    # plt.show()
    # print(data)
    return ds

# split the filename to get the sensor, target, temperature, abtime and num
def split_filename(filename):
    sensor = filename.split('-')[1]
    # target: TNT is label 0, 2,4-DNPA is label 1
    target = filename.split('-')[2]
    temperature = filename.split('-')[3]
    abtime = filename.split('-')[4]
    num = filename.split('-')[5].split('.')[0]
    return sensor, target, temperature, abtime, num

# transfer data to json
def insert_j_to_json(path, detec_target):
    samples_insert = []
    data_f = []
    filenames = readInfo(path)
    for filename in filenames:
        sensor, target, temperature, abtime, num = split_filename(filename)
        if sensor == 'f':
            data_f = get_content(path, filename)
            samples_b = dict()
            samples_b['id'] = str(num) + '-' + str(temperature) + '-' + str(abtime)
            samples_b['label'] = 1
            samples_b['type'] = target
            samples_b['f'] = data_f
            samples_insert.append(samples_b)

    with open('path_to_target.json', 'w', encoding='utf-8') as outfile:
        json.dump(samples_insert, outfile)
        # print('new json is writing...')
        outfile.write('\n')


if __name__ == "__main__":
    # define the detection target, where BYS means 2,4-DNPA
    detec_target = 'BYS'
    path = 'Path/to/BYS'
    print('')
    print('---------------------Transfer data to json----------------------------')
    insert_j_to_json(path, detec_target)
    print("\r", end="")
    sys.stdout.flush()
    time.sleep(0.05)