import pennylane as qml
import pandas as pd
from pennylane import numpy as np
qubitn=2
dev = qml.device("lightning.qubit", wires=range(qubitn))

def preFuncPair(weight, bias, data1, data2):
    t = np.tensordot(data1, weight,((0),(1)))-bias
    T = np.tensordot(data2, weight,((0),(1)))-bias

    layerN=t.shape[0]
    for j in range(layerN-1):
        for i in range(t.shape[1]//2):
            qml.RX(t[j,2*i],wires=i)
            qml.RZ(t[j,2*i+1],wires=i)
        for i in range(t.shape[1]//2-1):
            qml.CNOT(wires=[i,i+1])
    for i in range(t.shape[1]//2):
        qml.RX(t[layerN-1,2*i],wires=i)
        qml.RZ(t[layerN-1,2*i+1],wires=i)

    for j in range(layerN-1,0,-1):
        for i in range(t.shape[1]//2-1,-1,-1):
            qml.RZ(-T[j,2*i+1],wires=i)
            qml.RX(-T[j,2*i],wires=i)
        for i in range(t.shape[1]//2-2,-1,-1):
            qml.CNOT(wires=[i,i+1])
    for i in range(t.shape[1]//2-1,-1,-1):
        qml.RZ(-T[0,2*i+1],wires=i)
        qml.RX(-T[0,2*i],wires=i)


def preFunc(weight, bias, data):
    t = np.tensordot(data, weight,((0),(1)))-bias

    layerN=t.shape[0]
    for j in range(layerN-1):
        for i in range(t.shape[1]//2):
            qml.RX(t[j,2*i],wires=i)
            qml.RZ(t[j,2*i+1],wires=i)
        for i in range(t.shape[1]//2-1):
            qml.CNOT(wires=[i,i+1])
    for i in range(t.shape[1]//2):
        qml.RX(t[layerN-1,2*i],wires=i)
        qml.RZ(t[layerN-1,2*i+1],wires=i)


def postfunc(params):
    N=len(params)//(qubitn*2)
    for j in range(N):
        for i in range(qubitn):
            qml.RX(params[i*2],wires=i)
            qml.RZ(params[i*2+1],wires=i)
        for i in range(qubitn-1):
            qml.CNOT(wires=[i,i+1])

@qml.qnode(dev)
def preCircuitTest(weight, bias, data):
    preFunc(weight, bias, data)
    return qml.state()

@qml.qnode(dev)
def preCircuitPair(weight, bias, data1, data2):
    preFuncPair(weight, bias, data1, data2)
    return qml.probs(wires=range(qubitn))

@qml.qnode(dev)
def fullCircuit(weight,bias,postParams,data):
    preFunc(weight,bias,data)
    postfunc(postParams)
    return qml.probs(wires=range(qubitn))

def preCostFunction(weight,bias,dataSetsX,dataSetsY):
    loss=0
    for i in range(dataSetsX.shape[0]):
        makeResult = preCircuitPair(weight, bias, dataSetsX[i, 0:4], dataSetsX[i, 4:8])[0]
        loss += makeResult + (1-2*dataSetsY[i]) * makeResult
    return loss/dataSetsX.shape[0]

def postCostFunction(weight,bias,postParams,dataSetsX,dataSetsY):
    loss=0
    for i in range(dataSetsX.shape[0]):
        prob1 = fullCircuit(weight, bias, postParams, dataSetsX[i, 0:4])
        prob2 = fullCircuit(weight, bias, postParams, dataSetsX[i, 4:8])
        if (dataSetsY[i]==0):
            loss += np.inner(prob1, prob2)
        else:
            loss += np.sum((prob1-prob2)**2)
    return loss/dataSetsX.shape[0]

def preCostFunctionTest(weight,bias,dataSets):
    loss=0
    for i in range(dataSets.shape[0]):
        makeResult = np.abs(np.vdot(preCircuitTest(weight, bias, dataSets[i, 0:4]),preCircuitTest(weight, bias, dataSets[i, 4:8])))**2
        loss += makeResult
    return loss/dataSets.shape[0]


def preTrain(cost,weight,bias,pairs,index,minBatchSize,datan,steps,batchLoop):
    for i in range(steps):
#        batch_size=minBatchSize*(i+1)
        batch_size=minBatchSize
        for j in range(batchLoop):
            if (batch_size*(j+1)>=datan):
                batchX = np.array(pairs[batch_size*j:datan],requires_grad=False)
                batchY = np.array(index[batch_size*j:datan],requires_grad=False)
            else:
                batchX = np.array(pairs[batch_size*j:batch_size*(j+1)],requires_grad=False)
                batchY = np.array(index[batch_size*j:batch_size*(j+1)],requires_grad=False)
            weight, bias, _, _ = optimizer.step(cost, weight, bias, batchX, batchY)
            print(i,j)
            file=open('saveWeight.txt','wt')
            file.write('[')
            for i1 in range(weight.shape[0]):
                file.write('[')
                for j1 in range(weight.shape[1]):
                    file.write('[')
                    for k1 in range(weight.shape[2]-1):
                        file.write(str(weight[i1,j1,k1])+',')
                    file.write(str(weight[i1,j1,weight.shape[2]-1])+']')
                    if (j1<weight.shape[1]-1):
                        file.write(',')
                    else:
                        file.write(']')
                if (i1<weight.shape[0]-1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            file=open('saveBias.txt','wt')
            file.write('[')
            for i1 in range(bias.shape[0]):
                file.write('[')
                for j1 in range(bias.shape[1]):
                    file.write(str(bias[i1,j1]))
                    if (j1<bias.shape[1]-1):
                        file.write(',')
                    else:
                        file.write(']')
                if (i1<bias.shape[0]-1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            print(cost(weight, bias, batchX, batchY))
    return weight,bias

def postTrain(cost,weight,bias,postParam,pairs,index,minBatchSize,datan,steps,batchLoop):
    weight=np.array(weight,requires_grad=True)
    bias=np.array(bias,requires_grad=True)
    for i in range(steps):
 #       batch_size=minBatchSize*(i+1)
        batch_size=minBatchSize
        for j in range(batchLoop):
            if (batch_size*(j+1)>=datan):
                batchX = np.array(pairs[batch_size*j:datan],requires_grad=False)
                batchY = np.array(index[batch_size*j:datan],requires_grad=False)
            else:
                batchX = np.array(pairs[batch_size*j:batch_size*(j+1)],requires_grad=False)
                batchY = np.array(index[batch_size*j:batch_size*(j+1)],requires_grad=False)
            if (j==0): print(cost(weight, bias, postParam, batchX, batchY))
            weight, bias, postParam, _, _ = optimizer.step(cost, weight, bias, postParam, batchX, batchY)
            print(i,j)
            file=open('savePost.txt','wt')
            file.write('[')
            for i1 in range(postParam.shape[0]):
                file.write(str(postParam[i1]))
                if (i1<postParam.shape[0]-1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            file=open('saveWeight.txt','wt')
            file.write('[')
            for i1 in range(weight.shape[0]):
                file.write('[')
                for j1 in range(weight.shape[1]):
                    file.write('[')
                    for k1 in range(weight.shape[2]-1):
                        file.write(str(weight[i1,j1,k1])+',')
                    file.write(str(weight[i1,j1,weight.shape[2]-1])+']')
                    if (j1<weight.shape[1]-1):
                        file.write(',')
                    else:
                        file.write(']')
                if (i1<weight.shape[0]-1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            file=open('saveBias.txt','wt')
            file.write('[')
            for i1 in range(bias.shape[0]):
                file.write('[')
                for j1 in range(bias.shape[1]):
                    file.write(str(bias[i1,j1]))
                    if (j1<bias.shape[1]-1):
                        file.write(',')
                    else:
                        file.write(']')
                if (i1<bias.shape[0]-1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            print(cost(weight, bias, postParam, batchX, batchY))
    return weight,bias,postParam



optimizer = qml.AdamOptimizer()
steps = 100
minBatchSize = 150
testBatchSize = 150
batchLoop = 1
preLayerN=1
postLayerN=1
weight = np.random.randn(preLayerN,4,qubitn*2)
bias = np.random.randn(preLayerN,qubitn*2)
postParam = np.random.randn(qubitn*2*postLayerN)

#weight= np.array([[[0.19921960139562034,-0.8141362018379946,-0.3824479057615128,-0.026130048275873556],[-1.0669272726551702,-0.10811183426084939,0.11309280273317072,0.10423813924373199],[1.0203611844404998,-0.5277103065825924,-0.6822353355685846,0.9833052984360312],[1.9770207250266705,0.814262634421346,-0.5740454998740312,-0.30858527273732433]]])
#bias = np.array([[1.4030607788877592,0.2107559705477868,-1.0152523461969556,-0.9582990507480067]])
#postParam = np.array([0.34532646151066665,-0.28191673626004743,0.00021834544850138043,0.24457949204496574])


preData = pd.read_csv('simple_iris.csv').iloc[:,1:].to_numpy()
postData = pd.read_csv('simple_iris.csv').iloc[:,1:].to_numpy()
prePairs = np.array(preData[:, 0:8],requires_grad=False)
preIndex = np.array(preData[:, 8],requires_grad=False)
print(preCostFunction(weight, bias, prePairs, preIndex))
print(postCostFunction(weight, bias, postParam, prePairs, preIndex))

#weight,bias=preTrain(preCostFunction,weight,bias,prePairs,preIndex,minBatchSize,preDatan,steps,batchLoop)

#getSample=np.random.randint(0, postDatan, (testBatchSize,))
#preBatchX = prePairs[getSample]
#preBatchY = preIndex[getSample]
#getSample=np.random.randint(0, postDatan, (testBatchSize,))
#postBatchX = postPairs[getSample]
#postBatchY = postIndex[getSample]


for i in range(10):
    
    preData = pd.read_csv('simple_iris.csv').iloc[:,1:].to_numpy()
    postData = pd.read_csv('simple_iris.csv').iloc[:,1:].to_numpy()
    prePairs = np.array(preData[:, 0:8],requires_grad=False)
    preIndex = np.array(preData[:, 8],requires_grad=False)
    postPairs = np.array(preData[:, 0:8],requires_grad=False)
    postIndex = np.array(preData[:,8],requires_grad=False)
    preDatan = prePairs.shape[0]
    postDatan = postPairs.shape[0]
    
    weight,bias,postParam=postTrain(postCostFunction,weight,bias,postParam,postPairs,postIndex,minBatchSize,postDatan,steps,batchLoop)


    preData = pd.read_csv('pre_iris.csv').iloc[:,1:].to_numpy()
    postData = pd.read_csv('post_iris.csv').iloc[:,1:].to_numpy()
    prePairs = np.array(preData[:, 0:8],requires_grad=False)
    preIndex = np.array(preData[:, 8],requires_grad=False)
    postPairs = np.array(preData[:, 0:8],requires_grad=False)
    postIndex = np.array(preData[:,8],requires_grad=False)

    preBatchX = prePairs
    preBatchY = preIndex
    postBatchX = postPairs
    postBatchY = postIndex
    print(preCostFunction(weight, bias, preBatchX, preBatchY))
    print(postCostFunction(weight, bias, postParam, postBatchX, postBatchY))

