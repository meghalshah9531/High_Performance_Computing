# High Performance Computing
In this programming project, an interaction of n particles throgh a gravitational force is simulated. Resulting from the interaction each particle occupies new positions. Calculation is done by Newton's universal law of gravitation. Usually, sequential programming is really computationally expensive and time consuming. This problem can be solved by OpenMPI, which is an open source Massage Passing Interface. It is basically a communication between two or more processors on a network. Comunication type could be point-to-point communication (means only two processors communicate at time), or collective communication (one or more processors send/receive information in parallel at the same time). 

First of all, user must define total number of particles in the simulation box (3D problem), time step (delta_t), and total number of steps to be calculated or end time of the simulation. 


# Scalability Tests  
Scalability is widely used to indicate the ability of hardware and software to deliver greater computational power when the amount of resources is increased [[Source](https://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling/)]. 

## Strong Scaling 
![Table](https://github.com/meghalshah9531/High_Performance_Computing/blob/main/Images/Strong_Scaling.PNG) ![Graph](https://github.com/meghalshah9531/High_Performance_Computing/blob/main/Images/Graph_Strong_Scaling.PNG)


## Weak Scaling 
![Table](https://github.com/meghalshah9531/High_Performance_Computing/blob/main/Images/Weak_Scaling.PNG) ![Graph](https://github.com/meghalshah9531/High_Performance_Computing/blob/main/Images/Graph_Weak_Scaling.PNG)

