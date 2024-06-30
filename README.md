# CivRealm diplomacy trading
![image](https://github.com/spidermonk7/CivRealm_diplomacy_trading/assets/98212025/a9793e13-208d-4459-b1e4-069510185801)

This function focus on solving the Techs Trading Tasks based on CivRealm Environment: [https://github.com/bigai-ai/civrealm]
And in this file, the mian codes are used to train a **Double DQN model** to solve the task. However, till now it performs poorly. 
We'll try to fix it later!!!


## Quickstart
We've offered two different models: MLP and Vanilla CNN for QNet, and it's optional whether to train w/o priority replay and whether to encode actions with autoencoder. 
For example, to train a CNN Qnet with auto-encoder and w/o priority:
    
    python run.py --mode=train --model_type=CNN --pr=False --auto_encoder=True

And to test the model's performance, run:
   
    python run.py --mode=valid --model_type=CNN --pr=False --auto_encoder=True


## Arguments
Here we list out some optional arguments F.Y.I
| Arguments      | default | --help     |
| :---:        |    :----:   |          :---: |
| --model_type      | MLP       | MLP or CNN as Qnet   |
| --pr   | True        | if use priority replay or not      |
| --train_episodes   | 100        | episode numbers for training      |
| --valid_episodes   |  1       | episode numbers for validation      |
| --difficulty   | easy        | minigame difficulty level      |
| --seed   | 0        | random seed      |
| --Tmax   | 10        | max steps of each episodes      |
| --mode   | train        | train or test      |
| --auto_encoder   | True        | if use autoencoder or not      |
