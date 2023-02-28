import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
np.random.seed(0)

# Load the Iris dataset from a CSV file
iris_data = pd.read_csv('iris.csv')

# Extract the features (inputs) from the dataset
X = iris_data.iloc[:, 0:-1]

# Extract the labels (outputs) from the dataset
y = iris_data.iloc[:, -1]

# Normalize the features using the StandardScaler method
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
print(f"Normalizing with mean={scaler.mean_},std={scaler.scale_}")

X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

N=150*50
cnt=0
data = X_normalized.to_numpy()
pair_data = np.zeros((N, 8))
for i in range(150):
    for j in range(i+1,150):
        same = 1 if (i//50==j//50) else 0
        if (same==1): continue
        pair_data[cnt][:4] = data[i,:]
        pair_data[cnt][4:8] = data[j,:]
        cnt=cnt+1

pair_iris = pd.DataFrame(pair_data, columns=["A1","A2","A3","A4","B1","B2","B3","B4"])

pair_iris.to_csv("pre_iris.csv")


N=150*149//2
cnt=0
data = X_normalized.to_numpy()
pair_data = np.zeros((N, 9))
for i in range(150):
    for j in range(i+1,150):
        same = 1 if (i//50==j//50) else 0
        pair_data[cnt][:4] = data[i,:]
        pair_data[cnt][4:8] = data[j,:]
        pair_data[cnt][8] = same
        cnt=cnt+1

pair_iris = pd.DataFrame(pair_data, columns=["A1","A2","A3","A4","B1","B2","B3","B4","label"])
pair_iris.to_csv("post_iris.csv")
