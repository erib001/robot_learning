# Overview
Implemented different additions to the plain SAC algorithm in order to achieve best multi-task policy. 
• One-hot encoding,
• Disentangled Alphas (DA),
• Project conflicting Gradients (PCGrad),
• Curriculum Learning (CL), and
• Multi-head architecture for the Actor as well as the Critics

More details in [Summary Paper](https://github.com/erib001/robot_learning/blob/master/Robot_Learning.pdf)

# Training

All the parameter files for each method can be found in the `param` folder.

To start training use the following command

`python -W ignore train.py mt3_base`

with the correct parameter file.

# Play

To evaluate the best model the following command can be used

`python play.py mt3_base`

with the correct parameter file.
