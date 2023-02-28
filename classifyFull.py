import pennylane as qml
import pandas as pd
from pennylane import numpy as np
qubitn=3
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
        temp = np.inner(prob1, prob2)/np.sum(prob2**2)**0.5/np.sum(prob1**2)**0.5
        loss += temp + (1-2*dataSetsY[i])*temp
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
                batchY = np.array(index[batch_size*j:datan],requires_grad=False)
            weight, bias, batchX, batchY = optimizer.step(cost, weight, bias, batchX, batchY)
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

def postTrain(cost,weight,bias,postParam,pairs,index,minBatchSize,datan,steps,batchLoop):
    weight=np.array(weight,requires_grad=False)
    bias=np.array(bias,requires_grad=False)
    for i in range(steps):
 #       batch_size=minBatchSize*(i+1)
        batch_size=minBatchSize
        for j in range(batchLoop):
            if (batch_size*(j+1)>=datan):
                batchX = np.array(pairs[batch_size*j:datan],requires_grad=False)
                batchY = np.array(index[batch_size*j:datan],requires_grad=False)
            else:
                batchX = np.array(pairs[batch_size*j:batch_size*(j+1)],requires_grad=False)
                batchY = np.array(index[batch_size*j:datan],requires_grad=False)
            _, _, postParam, batchX, batchY = optimizer.step(cost, weight, bias, postParam, batchX, batchY)
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
            print(cost(weight, bias, postParam, batchX, batchY))


preData = pd.read_csv('pre_iris.csv').iloc[:,1:].to_numpy()
postData = pd.read_csv('post_iris.csv').iloc[:,1:].to_numpy()
prePairs = np.array(preData[:, 0:8],requires_grad=False)
preIndex = np.array(preData[:, 8],requires_grad=False)
postPairs = np.array(preData[:, 0:8],requires_grad=False)
postIndex = np.array(preData[:,8],requires_grad=False)

optimizer = qml.AdagradOptimizer()
preDatan = prePairs.shape[0]
postDatan = postPairs.shape[0]
steps = 100
minBatchSize = 50
testBatchSize = 1000
batchLoop = 1
preLayerN=3
postLayerN=5
weight = np.random.randn(preLayerN,4,qubitn*2)
bias = np.random.randn(preLayerN,qubitn*2)
postParam = np.random.randn(qubitn*2*postLayerN)
#weight = np.array([[[0.27853980364754943,-0.6920209305237355,-0.2827156942384427,0.07698712779538429,0.03374446348944275,-0.8829822317737085],[-0.13489232808783605,-1.0698144255848125,0.48748144055452464,1.6777524535393993,1.288782582515073,-1.3329668831195132],[-0.06923741625836913,-0.9131643037600264,-0.9136381826593464,0.16805936609509073,0.2511325242414259,0.6266715966203307],[-0.7311291211926151,1.248553226214355,-1.716763836118665,2.0882108507016515,0.5814083471526671,-0.3992825718240215]],[[0.2799815621994025,-0.31892351281818976,1.128328887038785,1.3657433560439654,0.5443693425387209,0.7086229488454364],[-1.2971723977000733,1.2036195083631873,-0.14267847997911637,0.7757474787628352,-0.47190935940743517,0.5381758793250879],[0.4809857282965339,-0.9410530966449303,0.4047370611363386,0.9383167632764846,-1.0889778264870622,0.9558599671213411],[0.03890617211941592,-0.04934554441660951,3.035275863022547,1.3450627015205514,-2.1775982449817075,-0.1457345710000091]],[[0.3371242716376434,-0.21494455460253664,0.7276641521973823,0.0641946608536838,-1.1875799915097316,-0.8626579492161321],[0.3999359461008985,0.7069404694530306,0.8793528092962832,-1.0613923171934492,-0.9345092073397875,0.20131868233263273],[-1.0785611701187134,-1.1312585027373905,-0.45089253655862216,1.5682692310216364,-1.7574431968285356,1.3787145175854418],[0.7388606846773916,-0.3630301429326001,-1.8399723065725901,-0.5868702482071101,2.2037310063452162,-0.4591921107577075]]])
#bias = np.array([[0.8196677430971514,-0.8238497120614565,0.8584789598445245,0.2812953802667046,1.09160041070184,-0.7583453919351755],[-1.625492353084568,-1.1734650177225792,-1.601175251630768,-0.3846155865137127,0.5078664665049263,0.32050876532856465],[-0.06787999905470525,1.3949084440984625,-0.017295867094159555,-1.1884255354533475,-1.6195029568284673,0.1239043395823184]])
#postParam = np.array([0.5349930777400164,-1.3008440677810804,-0.5467252707337062,0.6045628582162251,-0.18028893455297237,-0.856647521715801,-0.44914886662345604,-0.7594230759517939,0.6320902557500857,-0.4101349636626384,-0.12624720217717364,0.23611551663039415,0.22435304607394552,1.2362119233894446,0.4861724130135441,1.0273249499570194,1.1891935485777947,0.40996898996276016,-2.2120127339667213,-0.4869621036904792,-0.9126613991135377,-0.15822843785499852,-0.34186686467527744,-0.6248203578613069])

preTrain(preCostFunction,weight,bias,prePairs,preIndex,minBatchSize,preDatan,steps,batchLoop)
postTrain(postCostFunction,weight,bias,postParam,postPairs,postIndex,minBatchSize,postDatan,steps,batchLoop)

getSample=np.random.randint(0, postDatan, (testBatchSize,))
preBatchX = prePairs[getSample]
preBatchY = preIndex[getSample]
getSample=np.random.randint(0, postDatan, (testBatchSize,))
postBatchX = postPairs[getSample]
postBatchY = postIndex[getSample]
print(preCostFunction(weight, bias, preBatchX, preBatchY))
print(postCostFunction(weight, bias, postParam, postBatchX, postBatchY))
