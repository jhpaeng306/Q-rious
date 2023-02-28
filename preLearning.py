import pennylane as qml
import pandas as pd
from pennylane import numpy as np

qubitn = 3
dev = qml.device("lightning.qubit", wires=range(qubitn))


# Data에 따라 달라지는 unitary operator를 만드는 circuit입니다
def preFunc(weight, bias, data):
    t = np.tensordot(data, weight, ((0), (1))) - bias

    layerN = t.shape[0]
    for j in range(layerN - 1):
        for i in range(t.shape[1] // 2):
            qml.RX(t[j, 2 * i], wires=i)
            qml.RZ(t[j, 2 * i + 1], wires=i)
        for i in range(t.shape[1] // 2 - 1):
            qml.CNOT(wires=[i, i + 1])
    for i in range(t.shape[1] // 2):
        qml.RX(t[layerN - 1, 2 * i], wires=i)
        qml.RZ(t[layerN - 1, 2 * i + 1], wires=i)


# 두개의 Data에 대해 만들어지는 wavefucntion이 서로 직교하는 지 확인하는 circuit입니다.
# 0000000이 측정될 확률이 내적값의 크기가 됩니다.
def preFuncPair(weight, bias, data1, data2):
    t = np.tensordot(data1, weight, ((0), (1))) - bias
    T = np.tensordot(data2, weight, ((0), (1))) - bias

    layerN = t.shape[0]
    for j in range(layerN - 1):
        for i in range(t.shape[1] // 2):
            qml.RX(t[j, 2 * i], wires=i)
            qml.RZ(t[j, 2 * i + 1], wires=i)
        for i in range(t.shape[1] // 2 - 1):
            qml.CNOT(wires=[i, i + 1])
    for i in range(t.shape[1] // 2):
        qml.RX(t[layerN - 1, 2 * i], wires=i)
        qml.RZ(t[layerN - 1, 2 * i + 1], wires=i)

    for j in range(layerN - 1, 0, -1):
        for i in range(t.shape[1] // 2 - 1, -1, -1):
            qml.RZ(-T[j, 2 * i + 1], wires=i)
            qml.RX(-T[j, 2 * i], wires=i)
        for i in range(t.shape[1] // 2 - 2, -1, -1):
            qml.CNOT(wires=[i, i + 1])
    for i in range(t.shape[1] // 2 - 1, -1, -1):
        qml.RZ(-T[0, 2 * i + 1], wires=i)
        qml.RX(-T[0, 2 * i], wires=i)


# 내적값을 줍니다
@qml.qnode(dev)
def preCircuitPair(weight, bias, data1, data2):
    preFuncPair(weight, bias, data1, data2)
    return qml.probs(wires=range(qubitn))


# 전체 회로입니다
@qml.qnode(dev)
def fullCircuit(weight, bias, postParams, data):
    preFunc(weight, bias, data)
    postfunc(postParams)
    return qml.probs(wires=range(qubitn))


# 내적값을 기반으로한 cost function입니다
def preCostFunction(weight, bias, dataSetsX, dataSetsY):
    loss = 0
    for i in range(dataSetsX.shape[0]):
        makeResult = preCircuitPair(weight, bias, dataSetsX[i, 0:4], dataSetsX[i, 4:8])[0]
        loss += makeResult + (1 - 2 * makeResult) * dataSetsY[i]
    return loss / dataSetsX.shape[0]



# 내적값을 바탕으로 train하는 부분입니다
def preTrain(cost, weight, bias, pairs, index, minBatchSize, datan, steps, batchLoop):
    batch_size=datan
    for i in range(steps):
        j=0
        if (batch_size * (j + 1) >= datan):
            batchX = np.array(pairs[batch_size * j:datan], requires_grad=False)
            batchY = np.array(index[batch_size * j:datan], requires_grad=False)
        else:
            batchX = np.array(pairs[batch_size * j:batch_size * (j + 1)], requires_grad=False)
            batchY = np.array(index[batch_size * j:batch_size * (j + 1)], requires_grad=False)
        print(i, j)
        weight, bias, _, _ = optimizer.step(cost, weight, bias, batchX, batchY)
        file = open('saveWeight.txt', 'wt')
        file.write('[')
        for i1 in range(weight.shape[0]):
            file.write('[')
            for j1 in range(weight.shape[1]):
                file.write('[')
                for k1 in range(weight.shape[2] - 1):
                    file.write(str(weight[i1, j1, k1]) + ',')
                file.write(str(weight[i1, j1, weight.shape[2] - 1]) + ']')
                if (j1 < weight.shape[1] - 1):
                    file.write(',')
                else:
                    file.write(']')
            if (i1 < weight.shape[0] - 1):
                file.write(',')
            else:
                file.write(']')
        file.close()
        file = open('saveBias.txt', 'wt')
        file.write('[')
        for i1 in range(bias.shape[0]):
            file.write('[')
            for j1 in range(bias.shape[1]):
                file.write(str(bias[i1, j1]))
                if (j1 < bias.shape[1] - 1):
                    file.write(',')
                else:
                    file.write(']')
            if (i1 < bias.shape[0] - 1):
                file.write(',')
            else:
                file.write(']')
        file.close()
        print('loss',cost(weight, bias, batchX, batchY))
    return weight, bias


optimizer = qml.AdamOptimizer()
steps = 25
minBatchSize = 150
testBatchSize = 150
batchLoop = 1
preLayerN = 4
weight = np.random.randn(preLayerN, 4, qubitn * 2)
bias = np.random.randn(preLayerN, qubitn * 2)


for i in range(10):
    preData = pd.read_csv('simple_iris.csv').iloc[:, 1:].to_numpy()
    Pairs = np.array(preData[:, 0:8], requires_grad=False)
    Index = np.array(preData[:, 8], requires_grad=False)
    Datan = Pairs.shape[0]
    weight,bias=preTrain(preCostFunction,weight,bias,Pairs,Index,minBatchSize,Datan,steps,batchLoop)

