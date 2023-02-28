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


# data에 의존하지 않는 부분입니다
# 물리적인 역할은 직교하는 wavefunction들을 측정하기 용이한 wavefunction들로 대응시켜주는 역할입니다.
def postfunc(params):
    N = len(params) // (qubitn * 2)
    for j in range(N):
        for i in range(qubitn):
            qml.RX(params[j * 2 * qubitn + i * 2], wires=i)
            qml.RZ(params[j * 2 * qubitn + i * 2 + 1], wires=i)
        for i in range(qubitn - 1):
            qml.CNOT(wires=[i, i + 1])

# 전체 회로입니다
@qml.qnode(dev)
def fullCircuit(weight, bias, postParams, data):
    preFunc(weight, bias, data)
    postfunc(postParams)
    return qml.probs(wires=range(qubitn))


# 최종적인 확률분포를 이용한 cost function입니다
def postCostFunction(weight, bias, postParams, dataSetsX):
    loss = 0
    for i in range(dataSetsX.shape[0]):
        prob1 = fullCircuit(weight, bias, postParams, dataSetsX[i, 0:4])
        prob2 = fullCircuit(weight, bias, postParams, dataSetsX[i, 4:8])
        loss += np.inner(prob1, prob2)
    return loss / dataSetsX.shape[0]



# 최종적인 확률분포로 train을 하는 부분입니다
def postTrain(cost, weight, bias, postParam, pairs, minBatchSize, datan, steps, batchLoop):
    weight = np.array(weight, requires_grad=False)
    bias = np.array(bias, requires_grad=False)
    for i in range(steps):
        batch_size = minBatchSize
        for j in range(batchLoop):
            if (batch_size * (j + 1) >= datan):
                batchX = np.array(pairs[batch_size * j:datan], requires_grad=False)
            else:
                batchX = np.array(pairs[batch_size * j:batch_size * (j + 1)], requires_grad=False)
            if (j == 0): print(cost(weight, bias, postParam, batchX))
            _, _, postParam, _, = optimizer.step(cost, weight, bias, postParam, batchX)
            print(i, j)
            file = open('savePost.txt', 'wt')
            file.write('[')
            for i1 in range(postParam.shape[0]):
                file.write(str(postParam[i1]))
                if (i1 < postParam.shape[0] - 1):
                    file.write(',')
                else:
                    file.write(']')
            file.close()
            print('loss',cost(weight, bias, postParam, batchX))
    return weight, bias, postParam


optimizer = qml.AdamOptimizer()
steps = 100
minBatchSize = 3
testBatchSize = 3
batchLoop = 1
postLayerN = 5
postDatan = 3
postParam = np.random.randn(qubitn * 2 * postLayerN)

weight = [[[-0.30925210408936493,-0.007175905380512222,0.006595319419103151,-0.012935116413675049,0.12388541133249922,0.18573619435436026],[-0.10195125945251218,-0.003186710658489302,-0.006431718976443067,0.012411156928224394,0.14590322980221296,0.242823639046606],[1.410974316846665,0.004863397960644366,-0.03286001728344276,0.03701346644277903,-0.6251169942597914,-1.1893987058528774],[0.774519669753791,0.004844413159998628,-0.021036111852198714,-0.0038453447985330558,-0.48136696143379787,-0.7513537333348278]],[[0.30947562356191965,0.04418148107413806,-0.012832993373428712,0.023102590203107187,0.1266407049315765,0.03797298906020587],[0.09966986235181886,-0.08104138807338902,0.04873392774678778,0.1500669122634272,0.09281421078273434,0.0910755292134561],[-1.408969781922362,-0.01616865788628495,-0.15281643389818772,-0.3541013790295403,-0.6284737483078835,-0.16697991461123154],[-0.7700344020203888,-0.0006254692079218168,0.30529515151207975,-0.2553136646222347,-0.8098384118370571,-0.17862238681559556]],[[-0.0032416502821970552,-0.025409362051489144,0.10942882530553336,-0.001349315812765433,-0.05719955587134022,-0.07822149623753824],[-0.012829862908905825,0.013222556927441769,0.05251091455807551,-0.0942172547729733,-0.05585667482374196,-0.23101205242475342],[0.011920156937690995,-0.10398445137229544,-0.4973011164788991,0.32439140374820064,0.279612403923888,0.7394504701796882],[0.015408097373414273,-0.08206568851334214,-0.36089564847371336,0.3082632817965115,0.3859219913185011,0.5450014582119219]],[[0.0037897447953294276,0.06139984290870199,0.04379959065260546,-0.07078063108128983,0.0923569279395629,-0.06967891828369276],[0.008154608555532138,0.05265312939621007,0.10794013344016194,-0.18929674394609586,0.06963851319321336,-0.17379936873262608],[0.007743982427380597,-0.27755212516258715,-0.2304489302461011,0.4599669695093131,-0.2490345691561281,0.350979806339717],[0.00357383865102905,-0.04814708231289914,-0.17213803667227293,0.39775912570803595,-0.30070065918184197,0.385297300576474]],[[-0.0005345143256678297,0.014500712805298945,0.004685956471868107,0.001594794603588165,-0.02225390047649108,0.012108730491385195],[0.002571749005533985,0.008991012115643572,-0.007709691992027879,-0.004164161401141486,0.04067619299451154,-0.007088720637838152],[-0.0008717359766229967,0.01600128713279797,-0.007784102688509933,0.040888905106676005,-0.10453289898722903,0.03526140415140034],[0.0016135069386920605,0.040534402318802815,-0.006982772613226436,0.029806183802041405,-0.09783349837633541,0.03302120464369365]]]
bias = [[-0.3024209984598243,0.008939375520892355,1.5788211642991403,1.6156981205740526,1.3415215135085408,1.7030838681174965],[0.3040640022304194,0.4128844711618273,1.1119834152963788,1.2791188701725695,-1.20765991478133,1.1938335279619734],[0.03577337738083753,-0.05937239398468259,0.2148144915747455,-0.43967143244668566,-0.35677229311383196,0.049311740720275554],[-0.02284419200424428,-0.0028231980835837864,-0.8910556920208298,-0.009661486263705221,-0.8653736294929696,-0.017246762478867864],[-0.07086168231069778,-1.5659522802296256e-12,0.08604216709830485,1.7232333880982916e-11,-0.09714533424500316,3.241344067082717e-11]]
#postParam = [1.4650837418759923,1.4456760421915036,2.1845427385805936,1.8715634253659268,0.6512294271759379,1.935667287650052,-0.5456016371666685,1.2335442014181726,1.2374749548145378,0.933424029991937,2.1766450267975412,-0.9280792875582112]
postParam=np.array(postParam, requires_grad=True)

oneData = pd.read_csv('onedata_iris.csv').iloc[:, 1:].to_numpy()

postPairs = np.array([[oneData[49,i] for i in range(4)]+[oneData[50,i] for i in range(4)],
                     ])


for i in range(10):
    weight, bias, postParam = postTrain(postCostFunction, weight, bias, postParam, postPairs, minBatchSize, postDatan, steps, batchLoop)
