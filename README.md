
# Single shot Quantum Neural Network
#### QHack2023 Open Hackathon. Team name : Q-rious


## Project description
In our project, we built the Quantum Neural Network that works only with single shot. Our Quantum Neural Network classifies among $m$ labels, and is composed with two parts. 

### Encoder unitary gate : $U(\theta, w, b)$

First part encodes the input with appropriate weights ($w$) and bias ($b$) to construct an unitary gate working on $k$ qubits, where $2^k \geq m$. We refer such unitary matrix as $U(\theta, w, b)$ where $\theta$ is a parameter vector of given input data. We aim to construct this unitary matrix and $w, b$ to satisfy the conditions
- $U(\theta^A, w,b) U(\tilde{\theta}^A, w, b)=I$ where $\theta^A$ and $\tilde{\theta}^A$ are inputs with identical label $A$.
- 다를 경우에 어케 되야 하지....

When such conditions are satisfied, the unitary gate $U(\theta, w, b)$ represents the wave functions which are orthogonal at different labels, and identical at same labels. Thus, a function defined as a inner product between the wave functions of two input data, returns $1$ when their labels coincides, and returns $0$ when the labels differ. Note that only discriminating is sufficient to build a classifier since the ordering of the labels is irrelevant to the classified result.

To obtain the desired encoder matrix, the following is the steps we used.
1. 학습 방법 설명



### Decoder gate : $V(\lambda)$

Second part decodes the resulting qubits which is generated via encoder unitary gate. The desired result after the decoder gate is to have a qubit that matches the label. 
* 디코더가 하는 일 설명
* 디코더가 만족해야하는 성질 설명
* 디코더 학습 방법 설명.


## Key advantages of the Project
강점을 가지는 이유 2~3중 서술


Thus, our project shows advantages for the problems where
- situation 1
- situation 2

## Code description
preFunc function shows the structure of the quantum circuit. We followed the structure of EfficientSU2 in qiskit. 



## Power-Up plan
TBD


## File description

![initial](https://user-images.githubusercontent.com/124068470/221923133-2450187e-ae76-4525-a49e-5409a0a60a98.png)

preFunc : 
