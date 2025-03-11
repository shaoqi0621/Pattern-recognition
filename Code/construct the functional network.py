import pandas as pd
import numpy as np
import os
import math
import multiprocessing
from functools import partial
from time import time


def get_Matfile(path):
    path_list = os.listdir(path)
    files = [os.path.join(path, filename) for filename in path_list]
    return files


def read_csv(csvfilePath):
    data = pd.read_csv(csvfilePath, header=None)
    return data.values


def get_sum(num1, num2, array):
    return np.sum((array >= num1) & (array <= num2))


def Cal_allNeuronCrossMartix(response_Data, two, neuron_num):
    refData = two[:6000]
    refData = np.array(refData)
    refNum = int(two[-1])
    ref = refData[(0 < refData) & (refData <= 611)]

    singlecorr = np.zeros(neuron_num)
    if len(ref) == 0:
        return singlecorr

    for t in range(neuron_num):
        taget = response_Data[t, :]
        taget = taget[(0 < taget) & (taget <= 611)]
        if len(taget) == 0:
            singlecorr[t] = 0
        else:
            refspiketrain = np.zeros((len(ref), 50))
            rbegin = ref - 0.05
            bleft = rbegin[:, np.newaxis] + np.arange(50) * 0.002
            bright = rbegin[:, np.newaxis] + (np.arange(50) + 1) * 0.002
            for b in range(50):
                refspiketrain[:, b] = np.array([get_sum(bleft[i, b], bright[i, b], taget) for i in range(len(ref))])
            a = np.sum(refspiketrain, axis=0)
            c = math.sqrt(len(ref) * len(taget))
            a = a / c
            singlecorr[t] = np.max(a)

    return singlecorr


def Cal_FM(original_data, neuron_num):
    maxvalueTrain = original_data[:, :neuron_num]
    maxvalueTrain[maxvalueTrain == -1] = 0
    np.fill_diagonal(maxvalueTrain, 0)
    maxvalueTrain = maxvalueTrain.T + maxvalueTrain

    arr_mean = np.mean(maxvalueTrain)
    arr_std = np.std(maxvalueTrain)
    th1 = arr_mean + arr_std

    T1CM_Matrix = maxvalueTrain * (maxvalueTrain > th1)
    RM_Matrix = maxvalueTrain * (maxvalueTrain <= th1)

    TM = np.zeros((neuron_num, neuron_num))
    for r in range(neuron_num):
        rowValue = RM_Matrix[r, :]
        for c in range(neuron_num):
            if r != c:
                colValue = np.delete(rowValue, c)
                non_colValue = colValue[colValue.nonzero()]
                if len(non_colValue) > 0:
                    Mean = np.mean(non_colValue)
                    SD = np.std(non_colValue)
                    th = Mean + SD
                    TM[r, c] = th

    T2CM = RM_Matrix * (RM_Matrix >= TM)
    FM = T1CM_Matrix + T2CM

    return FM


def main(dataPath):
    response_matrix = read_csv(dataPath)
    neuron_num = response_matrix.shape[0]
    num = np.arange(neuron_num)
    itArr = np.column_stack((response_matrix, num))
    iterable = [tuple(row) for row in itArr]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # 调整 partial 函数的使用，只固定 neuron_num 参数
    func = partial(Cal_allNeuronCrossMartix, response_matrix, neuron_num=neuron_num)
    everowCoeff = pool.map(func, iterable)
    pool.close()
    pool.join()

    coeffMatrix = np.vstack(everowCoeff)
    return coeffMatrix, neuron_num


if __name__ == "__main__":
    begin_time = time()
    FilePath = "Your TimeMatrix File Path"
    savePath = "Your Saved File Path"
    FileName = get_Matfile(FilePath)
    for f in range(len(FileName)):
        each_file_name = FileName[f]
        name = os.path.basename(each_file_name)
        print(name)
        coeffMatrix, neuron_num = main(FileName[f])
        FM = Cal_FM(coeffMatrix, neuron_num)
        FM = pd.DataFrame(data=FM)
        FM.to_csv(savePath + '\\' + name, index=False, header=None)
    end_time = time()
    run_time = end_time - begin_time
    print('Running Time：', run_time)
