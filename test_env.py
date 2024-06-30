from civrealm.agents import RandomAgent
import gymnasium
from my_utils import *
from my_Agent import MyAgent
from civrealm.configs import fc_args
from tqdm import tqdm




fc_args['advisor'] = 'enabled' 

env = gymnasium.make('civrealm/FreecivMinitask-v0')
agent = MyAgent(debug=False)

done = False
step = 0

if __name__ == '__main__':
    succeed = 0
    episodes = 500
    for episode in tqdm(range(episodes), desc='Episodes'):
        observations, info = env.reset(
                                       minitask_pattern={
                                           'type':'diplomacy_trade_tech',
                                           'level':'easy',   
                                       }
                                       )
        done = False
        step = 0
        

        while True:
            print("actions are:", info['available_actions']['dipl'][1])
            exit()
            action = agent.random_act(observations, info)
            observations, reward, terminated, truncated, info = env.step(action)
            step += 1
            ours = env.civ_controller.player_ctrl.players[0]['inventions'][1:]
            print(f"techs: {ours}")
            # print(f'Step: {step}, Turn: {info["turn"]}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, action: {action}')
            # done = terminated or truncated

            if step == 10:
                print(f'Episode {episode} terminated, step reaches 10')
                break
            if info['minitask']['success']!=0 and info['minitask']['success']!=-1:
                print(f'Episode {episode} terminated, minitask succeed, the value is {info["minitask"]["success"]}')
                succeed += 1
                break       
         
        env.close()
        for item in info['mini_game_messages']:
            if 'msg' in item:
                print(item)
        # print(f"info = {info['mini_game_messages']}")
    # for message in info['mini_game_messages']:
    #     if 'msg' not in message:
    #         print(message['metrics'][0]['mini_score'], message['metrics'][0]['final_score'])
    # print(f"info_messages = {info['mini_game_messages']}")    
    print(f'Succeed rate = {succeed/episodes}')