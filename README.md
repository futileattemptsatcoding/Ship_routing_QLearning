# Ship_routing_QLearning

So far, I have written the code to create the final Q matrix, after which we can iterate over it using nearest neighbour algorithms to find the final path. The disadvantage of nearest neighbour algorithms in which future rewards or anti-rewards are not considered, is resolved if we use the Q matrix.


### Cases where this works:
When we have a straightforward path-finding case where there aren't any weird land obstacles, my agent reaches the goal state. This is the current situation after 1 episode:
![image](https://github.com/user-attachments/assets/729051cf-116e-40ba-8b0f-bf97e0cc8ff0)

I have initialized the Q matrix with the weights arranged as per a linear space of values increasing toward the goal in an exponential manner. This proves to be an adavantage when, near the start coordinate, the gradients do not move that drastically (less slope in gradient vs coordinate graph), and the nearer it moves towards the goal state, the increase in weights becomes more and more. 

![image](https://github.com/user-attachments/assets/a7e67cb0-64e6-44f8-b2b5-45d6f89fb61e)

### Cases where this doesn't work (To rectify soon):

Where there are land obstacles, the agent find reasons to get stuck (more distance towards ocean vs hitting the land) and thus cannot reach the goal state. 
