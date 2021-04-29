
# LSTM Variational auto encoder for dominant motion extraction.

This is the code accompanying paper:
[Dominant motion identification of multi-particle system using deep learning from video](https://arxiv.org/pdf/2104.12722.pdf)

## Introduction:

Extracting underlining dynamics of a system from data is a form of data-driven pattern recognition where similarities in the evolution of the system states overtime forms the basis to not only predict the future states of the system but also to identify and quantify the deviation of a systems from ideal state due to disturbance. One of the major challenges in distilling underlying dynamics from data is the availability of specific features which define the states of the system. Visually learning the dynamics of a system, say a group of ants foraging in space, in the form of a differential equation from videos of the systems not only allows for understanding of the system better but can also be used to transfer dynamics of this system to another system like swarm robots or optimization algorithms.

In this study we have provided a framework to extract trajectory of agents from a video of multi-agent system and distill the spatio-temporal trajectory information in the form of a differential equation. Figure below shows Schematic of framework. Frame in videos (a) are used to extract states(trajectory of particle) of the system at each time step (b). The extracted data is then fed into LSTM variational autoencoder (c). The network is trained till the reconstructed states or trajectories match the input states (d) and using sparse regression (e) the latent representation of states at each time steps is extracted is then used to model time differential equation of the system (f).
![alt text](https://github.com/BaratiLab/LSTM-VAE-for-dominant-motion-extraction/blob/main/img_util/ lstm_vae_v3.png
?raw=true) 

## LSTM VAE:
The time ordered spatial state of the system is fed into the LSTM VAE to encode the state of the system into a latent representation for each time-step.



TO DO:
1) Add LSTM VAE structure --> DONE
2) Add jupyter notebook for analysis --> DONE for ant data
3) Add weights --> DONE for ant data
4) Update readme project

![alt text](https://github.com/BaratiLab/LSTM-VAE-for-dominant-motion-extraction/blob/main/img_util/traj.gif?raw=true) 


