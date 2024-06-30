from civrealm.agents import RandomAgent
import gymnasium
from my_Agent import MyAgent
from my_utils import *
from tqdm import tqdm
from argparse import ArgumentParser
BUFFER_TRAIN_THRESH = 10
C = 10

env = gymnasium.make('civrealm/FreecivMinitask-v0')



# initial the Qnet
def run_training(episodes = 5000, difficulty = 'easy', seed = 0, Tmax = 10, model_type='CNN', pr = True, AE=None, Q_Act=None):
    agent = MyAgent(Q_Act, debug=False)
    losses = []
    K_train = 5
    buff = Buffer()
    p_max = 1
    for episode in tqdm(range(episodes), desc='Episodes'):
        observations, info = env.reset(seed=seed, 
                                       minitask_pattern={'type':'diplomacy_trade_tech', 
                                                         'level':'easy',})
        step=0
        done=False
        old_state = extract_state_from_env(env) 
        for step in range(Tmax):
            # carried out a step
            action = agent.act(old_state, info, observations=observations)
            observations, reward, terminated, truncated, info = env.step(action)
            new_state = extract_state_from_env(env)

            # store the trans
            if action is not None:
                buff.buffer.append((old_state, action, reward, new_state))
                if pr:
                    buff.p.append(p_max)
            
            # training.
            for k in range(K_train):
                # sample trans from data
                sampled_data, ids, weight = buff.sample(weighted=pr)
                # train the network model
                loss = agent.net_train(sampled_data, info, weight, model_type)
                # update the priority
                if pr:
                    p_max = loss
                    buff.update(ids, p_max)
                losses.append(loss)

                    
  
        # print(f'Episode: {episode}, Step: {step}, Turn: {info["turn"]}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, action: {action}')

            old_state = new_state

        # updata target network
        agent.updata_target()
        losses.append(loss)    
                
        print(f'Episode {episode} terminated with avg loss {loss}')
        Q_Act.save()
    env.close()
    return losses
    


def valid_model(model, episodes = 100, T=10,difficulty = 'easy'):
    done = False
    step = 0
    seed = 0
    succeed = 0
    agent = MyAgent(model, debug=False)

    
    for episode in tqdm(range(episodes), desc='Episodes'):
        observations, info = env.reset(seed=seed, 
                                       minitask_pattern={'type':'diplomacy_trade_tech', 
                                                         'level':'easy',})
        step=0
        done=False
        old_state = extract_state_from_env(env) 
        for step in range(T):
            # carried out a step
            action = agent.act(old_state, info, observations=observations)
            observations, reward, terminated, truncated, info = env.step(action)
            new_state = extract_state_from_env(env)
            old_state = new_state
            if info['minitask']['success']!=0 and info['minitask']['success']!=-1:
                # print(f'Episode {episode} terminated, minitask succeed, the value is {info["minitask"]["success"]}')
                succeed += 1
                break
        env.close()
    # print(f"success rate is {succeed/episodes}")
    return succeed/episodes



    
if __name__ == '__main__':
    
    args = ArgumentParser()
    args.add_argument('--model_type', type=str, default='MLP')
    args.add_argument('--pr', type=str, default='True')
    args.add_argument('--train_episodes', type=int, default=100)
    args.add_argument('--valid_episodes', type=int, default=1)
    args.add_argument('--difficulty', type=str, default='easy')
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--Tmax', type=int, default=10)
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--auto_encoder', type=str, default='True')
    
    args = args.parse_args()

    if args.pr == 'True':
        args.pr = True
    else:
        args.pr = False
    if args.auto_encoder == 'True':
        args.auto_encoder = True
    else:
        args.auto_encoder = False

    AE = None
    if args.auto_encoder:
        AE = AutoEncoder().to(device)
        AE.load_state_dict(torch.load('model/AutoEncoder.pth'))


    AEif = ''
    PRif = ''
    if args.auto_encoder:
        AEif = 'AE'
    if args.pr:
        PRif = 'PR'
    if args.mode == 'train':
        QNet = QNet(device=device, model_type=args.model_type, auto_encoder=AE, pr=args.pr)
        loss = run_training(episodes = args.train_episodes, difficulty = args.difficulty, seed = args.seed, 
                            Tmax = args.Tmax, model_type=args.model_type, pr = args.pr, AE=AE, Q_Act=QNet)
        np.save(f'losses/loss{args.model_type}_{AEif}_{PRif}.npy', loss)




    if args.mode == 'valid':
        model = QNet(device=device, model_type=args.model_type, auto_encoder=AE, pr=args.pr)
        path = 'model/QNet' + args.model_type + '_' + AEif + '_' + PRif + '.pth'
        model.model.load_state_dict(torch.load(path))
        succ_rate = valid_model(model, episodes = args.valid_episodes, T=10)
        print('-'*10  + '-'*10)
        print(f"Finish validating model: {path} for {args.valid_episodes} episodes, the success rate is {succ_rate}")
        # write the result to a file
        with open('results.txt', 'a') as f:
            f.write(f"Finish validating model: {path} for {args.valid_episodes} episodes, the success rate is {succ_rate}\n")
        print('-'*10  + '-'*10)