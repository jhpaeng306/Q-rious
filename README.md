
# Single shot Quantum Neural Network
#### QHack2023 Open Hackathon. Team name : Q-rious


## Project description
In our project, we built the Quantum Neural Network that works only with single shot. Our Quantum Neural Network classifies among $m$ labels, and is composed with two parts. 

First part encodes the input with appropriate weights ($w$) and bias ($b$) to construct an unitary gate working on $k$ qubits, where $2^k \geq m$. We refer such unitary matrix as $U(\theta, w, b)$ where $\theta$ is a parameter vector of given input data. We aim to construct this unitary matrix and $w, b$ to satisfy the conditions
- $U(\theta^A, w,b) U(\Tilde{\theta}^A, w, b)=I$ where $\theta^A$ and $\Tilde{\theta}^A$ are inputs with identical label $A$.
- 다를 경우에 어케 되야 하지....

When such conditions are satisfied, the unitary gate $U(\theta, w, b)$ represents the wave functions which are orthogonal at different labels, and identical at same labels. Thus, a function defined as a inner product between the wave functions of two input data, returns $1$ when their labels coincides, and returns $0$ when the labels differ. Note that only discriminating is sufficient to build a classifier since the ordering of the labels is irrelevant to the classified result.


Second part decodes the resulting qubits via encoder unitary gate. 

우리의 아이디어는 같은 label이면 함수값 1을, 그렇지 않으면 0을 출력하는 함수를 학습시키는 것이다.

The key ideas used in our project are
- idea 1
- idea 2
test

## Key advantages of the Project
강점을 가지는 이유 2~3중 서술


Thus, our project shows advantages for the problems where
- situation 1
- situation 2

## Power-Up plan
TBD


## File description

![initial](https://user-images.githubusercontent.com/124068470/221923133-2450187e-ae76-4525-a49e-5409a0a60a98.png)

preFunc : 
