
# Single shot Quantum Neural Network
#### QHack2023 Open Hackathon. Team name : Q-rious


## Project description
In our project, we built the Quantum Neural Network that works with only single shot by assigning wavefunction to lables. Our Quantum Neural Network classifies among $m$ labels, and is composed with two parts. 

![preFuncPair](https://user-images.githubusercontent.com/124068470/221951941-d04fc0b1-7f67-4172-b45e-ffd27cd129a9.png)

### Assigning the wavefunction : $U(x, w, b)$

The front part of our circuit assigns each case to quantum state that corresponds to the label, where $w$ and $b$ are fixed parameters and $x$ is parameter containing information about the input data. Which state is assigned to which label is automatically determined during the learning process. The circuit learns under the rule that the quantum state made by $U(x, w, b)$ should be orthogonal if the label of two datas are different and the size of inner product should be one if the label of two datas are same. Determining the size of innerproduct is quite easy, it is the probabilty of $\|000....00\rangle$ state for following circuit.


Under this rule, if two datas A, B are classified as the same label($A=B$) and two datas B,C are classified as the different label($B~=C$), A and C are automatically classified as different $A~=C$, since the quantum states corresponding to A and B only differ by the phase. This characteristics allows a new learning scheme. When making a classifier with neural network, originaly the output is assigned arbitrarily, without knowing which assignment is most natural. To avoid this problem we can change the loss function, by making loss function which becomes smaller when output for same label becomes similar and ouput for different label becomes different. However, this approach needs many training sets, because eventhough the neural net learns $A=B$ and $B\neq C$ it doesn't know $A\neq C$. For quantum neural network, especially when the network can classify by single shot, it is not a problem. For two wavefunctions to be distinguishable by the measurement, especially by the single shot, they should be orthogonal, which means for quantum classifier, wavefunction resulting from neural network should be orthogonal for different label. If we additionally give condition that the absolute value of the innerproduct between wavefucntions resulting from the same label becomes one, the number of dataset needed for learning sameness and difference can reduce dramatically, because when $A=B$ and $~B=C$, $~A=C$ hold for this scheme.

The ansatz for $U(x, w, b)$ is given as follows.
![preFunc](https://user-images.githubusercontent.com/124068470/221949350-a31aa87a-73ca-4cc5-b911-7732e592ed72.png)

### Measuring : $V(\lambda)$
Although the states are perfectly classified by assigning wavefunctions, we can't see the result. Appropriate measurement is needed to see the result. Any measurement is equivalent with applying a unitary transform and measuring qubits. The back part of our circuit finds this unitary transform. We use the same ansatz, with parameters independent with the data

Since the states with same labels are similar learning for whole dataset is not needed. Only one data from each label are needed to determine the parameters. Also, since the states with different labels are orthogonal, appropriate measurement that distinguishes label by a single shot should exist.



## Key advantages of the Project
For a classical computer, the learning need not be perfect, since the result can be determined if the crieteria is satisfied. However, for quantum neural network, if the result can not be determined by a single shot, there is always a chance for statistical noise to change the result. Inspired by the Qhack problems requiring discrimination by one shot measurement, which our team solved with this scheme, 

Thus, our project shows advantages for the problems where
- situation 1
- situation 2

## Code description
preFunc function shows the structure of the quantum circuit. We followed the structure of EfficientSU2 in qiskit. 


## File description








preFunc : 
