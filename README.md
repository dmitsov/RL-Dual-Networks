# RL-Dual-Networks
RL Course Project repository

How to use this project:
1. Create a folder in your google drive called DL-Agent
2. Upload all of the .py in the repository files and requirements_colab.txt to that folder
3. Upload the notebooks to Google Colab
4. Run one of them and see how it goes. While specifying in the runLearning method if the network should be a single stream or dueling network
(True for single stream and False for Dueling network). You have to pass a list of keys for which hyperparameters to use. They are defined in DL_Agent.py 
The different keys are: ['params_v0', 'params_v1', 'params_v2', 'params_v3', 'params_v4', 'params_v5',]

!!! Important: Please note that if you run one notepad after the other for the same parameter keys the results will be overwritten. So save them before running the notebooks
