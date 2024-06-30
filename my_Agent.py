from civrealm.agents import BaseAgent
import random
from my_utils import *


class MyAgent(BaseAgent):
    def __init__(self, Qnet=None, debug=False):
        super().__init__()
        # Qnet is a special model that can be trained(self defined class)
        self.Qnet = Qnet
        self.Qtarget = Qnet
        self.debug = debug


    def act(self, state, info, observations):
        available_actions = info['available_actions']['dipl'][1]
        
        if self.debug:
            for action in available_actions:
                print(action, available_actions[action], '\n')
            print(f"there are {len(available_actions)} possible actions")

        dipl_actor, dipl_action_dict = self.get_next_valid_actor(
            observations, info, 'dipl')
        # fc_logger.info(f'Valid actions: {dipl_action_dict}')
        if not dipl_actor or dipl_actor==2:
            return None
        
        available_actions = [action for action in dipl_action_dict.keys()]
        # print(f"the actions are {available_actions}")
        selected_action, _ = select_action(available_actions, self.Qnet, state, auto_encoder=self.Qnet.auto_encoder)

        return 'dipl', dipl_actor, selected_action


    def random_action_by_name(self, valid_action_dict, name):
        # Assume input actions are valid, and return a random choice of the actions whose name contains the input name.
        action_choices = [
            key for key in valid_action_dict.keys() if name in key]
        if action_choices:
            return random.choice(action_choices)
        else:
            return None



    def random_act(self, observations, info):
        dipl_actor, dipl_action_dict = self.get_next_valid_actor(
            observations, info, 'dipl')
        # fc_logger.info(f'Valid actions: {dipl_action_dict}')
        if not dipl_actor or dipl_actor==2:
            return None
        print(f"there are {len(dipl_action_dict)} possible actions")
        select_action = self.random_action_by_name(dipl_action_dict, '')

        # select_action = random.choice(list(dipl_action_dict.keys()))

        return 'dipl', dipl_actor, select_action


    def net_train(self, buffer_data, info, weight, model_type='MLP'):
        # buffer_data: [(state, action, reward, next_state), ...]
        # print(f"buffer_data = {buffer_data}, len = {len(buffer_data)}")
        state, action, reward, next_state = buffer_data
        loss = self.Qnet.train(
            state=state, action=action, reward=reward, next_state=next_state, 
            weight=weight, model_type=model_type, info=info, Q_target=self.Qtarget)
        return loss    
        
    def updata_target(self):
        self.Qtarget = self.Qnet