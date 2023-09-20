import random
import numpy as np
from collections import deque
# from nn_builder.pytorch.CNN import CNN
import torch
import pandas as pd
# from nn_builder.pytorch.NN import NN
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal
torch.autograd.set_detect_anomaly(True)
import math

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPSILON = 1e-6

torch.manual_seed(208)
torch.cuda.manual_seed(208)
torch.cuda.manual_seed_all(208)  # if you are using multi-GPU.
np.random.seed(208)  # Numpy module.
random.seed(208)  # Python random module.
torch.manual_seed(208)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
MAX_TIME_BUDGET = 1200 # the largest time budget among datasets in the AutoML challenge
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Env():
    def __init__(self, time_awareness, number_of_rows, number_of_columns,type_of_learning, t_0,
                dataset_meta_features, algorithms_meta_features, evaluate):
        self.env_name = "learning_curve"
        self.time_awareness = time_awareness
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.learning_curves = []
        self.freely_moving = True
        self.type_of_learning = type_of_learning
        self.t_0 = t_0
        self.num_dataset = 1
        self.evaluate = evaluate
        self.number_of_action_channels = 1
        self.nA = number_of_columns # Size of action space n*n
        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.list_datasets = sorted(self.dataset_meta_features.keys())
        self.list_algorithms = sorted(self.algorithms_meta_features.keys())
        if self.time_awareness:
            self.number_of_state_channels = 5
        else:
            self.number_of_state_channels = 2

    def get_A_star(self):
        return torch.argmax(self.s[0]).item()

    def reset(self):
        # self.dataset_name = dataset_name
        self.counters = {i:0.0 for i in range(len(self.list_algorithms))} # Counters keeping track of the time has been spent for each algorithm
        # dataset_meta_features = self.dataset_meta_features[dataset_name]
        self.total_time_budget = float(self.dataset_meta_features['time_budget'])
        self.remaining_time_budget = self.total_time_budget
        self.last_score = 0.0

        # Reset current state
        self.s = torch.ones(self.number_of_state_channels, self.number_of_rows, self.number_of_columns)
        if self.time_awareness:
            self.s[0] = (-1) * self.s[0] # Current validation performance: R_validation_C_A
            self.s[1] = 0.0 * self.s[1] # Time spent: C_A
            self.s[2] = self.remaining_time_budget * self.s[2]
            self.s[3] = (self.total_time_budget/MAX_TIME_BUDGET) * self.s[3]
            self.s[4] = 0.0 * self.s[4]
        else:
            self.s[0] = (-1) * self.s[0] # Current validation performance: R_validation_C_A
            self.s[1] = 0.0 * self.s[1] # Time spent: C_A
        return self.s

    def reveal(self, action, validation_learning_curves, test_learning_curves):
        A_star, A, delta_t = action
        done = False

        #=== The agent cannot exceed the total time budget
        if delta_t >= self.remaining_time_budget:
            delta_t = self.remaining_time_budget
            done = True

        #=== Reveal C_A_star, R_test_C_A_star
        if A_star==None:
            R_test_C_A_star, C_A_star = None, None
        else:
            R_test_C_A_star, C_A_star = test_learning_curves[self.list_algorithms[A_star]].get_last_point_within_delta_t(0, self.counters[A_star])

        #=== Reveal C_A, R_validation_C_A
        R_validation_C_A, C_A = validation_learning_curves[self.list_algorithms[A]].get_last_point_within_delta_t(delta_t, self.counters[A])

        #=== Update algorithm counters and the remaining time budget
        self.counters[A] = C_A
        self.remaining_time_budget = self.remaining_time_budget - delta_t

        #=== Prepare an observation to be sent to the agent
        observation = (A, C_A, R_validation_C_A)
        self.s[0,0,A] = R_validation_C_A
        self.s[1,0,A] = C_A/self.total_time_budget
        self.s[2] = self.remaining_time_budget/self.total_time_budget

        t = self.total_time_budget - self.remaining_time_budget
        normalized_t = np.log(1+t/self.t_0)/np.log(1+self.total_time_budget/self.t_0)
        reward = (R_validation_C_A-self.last_score) * (1-normalized_t)

        if math.isnan(reward):
            reward = 0.0
        self.last_score = R_validation_C_A
        return self.s, torch.tensor(reward), torch.tensor(done)

    def observe(self):
        """
          Return the current state
        """
        return self.s

class Agent():
    """ Double DQN agent
    """
    def __init__(self, number_of_algorithms):
        self.agent_name = 'DDQN'
        self.epsilon = 0.99 # For deciding exploitation or exploration
        self.epsilon_decay_rate = 0.95 # Epsilon is decayed after each episode with a fixed rate
        self.gamma = 0.99 # The weight for future rewards
        self.batch_size = 512
        self.memory = Replay(100000, self.batch_size)             # Experience Buffer
        self.number_of_state_channels = 5
        self.number_of_rows = 1
        self.number_of_columns = number_of_algorithms
        self.nA = 20
        self.seed = 208
        self.evaluate = False
        self.device = device
        self.main_dqn = DDQN_Network(in_channels=self.number_of_state_channels, num_actions=self.nA,
                                     hidden_dim1 = 32, hidden_dim2=16)
        self.target_dqn = DDQN_Network(in_channels=self.number_of_state_channels, num_actions=self.nA,
                                     hidden_dim1 = 32, hidden_dim2=16)

        # Send models to GPU
        self.main_dqn.to(self.device)
        self.target_dqn.to(self.device)

         # Optimizer and Loss function
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), lr=1e-3)
        self.mse = torch.nn.MSELoss()
        self.L1 = torch.nn.SmoothL1Loss()

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float)
            The last observation returned by the environment containing:
                (1) A: the explored algorithm,
                (2) C_A: time has been spent for A
                (3) R_validation_C_A: the validation score of A given C_A

        Returns
        ----------
        action : tuple of (int, int, float)
            The suggested action consisting of 3 things:
                (1) A_star: algorithm for revealing the next point on its test learning curve
                            (which will be used to compute the agent's learning curve)
                (2) A:  next algorithm for exploring and revealing the next point
                       on its validation learning curve
                (3) delta_t: time budget will be allocated for exploring the chosen algorithm in (2)

        Examples
        ----------
        >>> action = agent.suggest((9, 151.73, 0.5))
        >>> action
        (9, 9, 80)
        """
        ### TO BE IMPLEMENTED ###
        """
          Return an action to take based on epsilon (greedy or random action)
          :param state: the current state
          :return action: next action to take
        """
        if observation==None:
            A_star = None
        else:
            A_star = self.env.get_A_star()

        if self.evaluate:
            self.update_state(observation)

        random_number = np.random.uniform()
        if random_number < self.epsilon and not self.evaluate:
            # Random action
            A = torch.tensor(random.randint(0,self.env.nA-1))
        else:
            # Greedy action
            state = [self.env.s]
            state = torch.stack(state)
            state = state.to(self.device, dtype=torch.float)
            q_values = self.main_dqn(state)
            argmax = torch.argmax(q_values).item()
            A = torch.tensor(argmax)

        if self.env.s[4,0,A]==0:
            delta_t = 1.0
        else:
            delta_t = 2*float(self.env.s[4,0,A].item()*self.env.total_time_budget)

        print('delta_t=', delta_t)

        #Update remaining_time_budget
        if self.env.time_awareness:
            self.env.s[2] = (self.env.remaining_time_budget - delta_t)/self.env.total_time_budget
            self.env.s[4, 0, A] = delta_t/self.env.total_time_budget

        return (A_star, A.item(), delta_t)

    def update_state(self,observation):
        if observation!=None:
            A, C_A, R_validation_C_A = observation
            self.env.s[0,0,A] = R_validation_C_A
            self.env.s[1,0,A] = C_A/self.env.total_time_budget

    def train(self):
        """
        Train the network with a batch of samples
        :param states: The state before taking the action
        :param actions: action taken
        :param rewards: Reward for taking that action
        :param next_states: The state that the agent enters after taking the action
        :return loss: the loss value after training the batch of samples
        """
        if len(self.memory) >= self.batch_size:
            with torch.no_grad():
                states, actions, rewards, next_states, dones = self.memory.sample()

            # Send data to GPU
            states = torch.stack(states).to(self.device, dtype=torch.float)
            actions = torch.stack(actions).to(self.device, dtype=torch.float)
            rewards = torch.stack(rewards).to(self.device, dtype=torch.float)
            rewards = torch.reshape(rewards, (self.batch_size, 1))
            next_states = torch.stack(next_states).to(self.device, dtype=torch.float)
            dones = torch.stack(dones).to(self.device, dtype=torch.float)

            # Calculate target Q values using the Target Network
            selection = torch.argmax(self.main_dqn(next_states), dim = 1).unsqueeze(1)
            evaluation = self.target_dqn(next_states)
            evaluation = evaluation.gather(1, selection.long()) #size [256,1]

            # Calculte target
            target = rewards + evaluation*self.gamma

            # Calculate Q values using the Main Network
            if self.env.freely_moving:
                n_classes = self.env.number_of_action_channels * self.env.number_of_rows * self.env.number_of_columns
            else:
                n_classes = self.env.number_of_action_channels * 1 * self.env.nA
            n_samples = self.batch_size
            labels = torch.flatten(actions.type(torch.LongTensor), start_dim=0)
            labels_tensor = torch.as_tensor(labels)
            action_masks = torch.nn.functional.one_hot(labels_tensor, num_classes=n_classes).to(self.device, dtype=torch.float)
            q_value = action_masks * self.main_dqn(states)
            # Calculate loss
            loss = self.mse(target, q_value)
#             loss = self.L1(target, q_value)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Soft Copy the Main Network's weights to the Target Network
            self.soft_update_of_target_network(self.main_dqn, self.target_dqn,tau=5e-1)

            return loss
        return 0.0

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                'usage' : name of the competition
                'name' : name of the dataset
                'task' : type of the task
                'target_type' : target type
                'feat_type' : feature type
                'metric' : evaluatuon metric used
                'time_budget' : time budget for training and testing
                'feat_num' : number of features
                'target_num' : number of targets
                'label_num' : number of labels
                'train_num' : number of training examples
                'valid_num' : number of validation examples
                'test_num' : number of test examples
                'has_categorical' : presence or absence of categorical variables
                'has_missing' : presence or absence of missing values
                'is_sparse' : full matrices or sparse matrices

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of all algorithms

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'Meta-learningchallenge2022', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}

        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """

        ### TO BE IMPLEMENTED ###
        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.env = Env(True, 1, self.number_of_columns, 'any_time_learning', 60,
                    dataset_meta_features, algorithms_meta_features, self.evaluate)

        ########-------------------##########
        state = self.env.reset()

    def meta_train(self, datasets_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent with the validation and test learning curves

        Parameters
        ----------
        datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms

        validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['Erik']
        {'name':'Erik', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'Erik' :

        >>> validation_learning_curves['Erik']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['Erik']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['Erik']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """

        ### TO BE IMPLEMENTED ###
        print("########## " + self.agent_name + " is running ##########")
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

        for i in range(1):
            for dataset_name in datasets_meta_features:

                print('dataset_name = ', dataset_name)
                self.reset(datasets_meta_features[dataset_name], algorithms_meta_features)
                # self.env = Env(True, 1, self.number_of_columns, 'any_time_learning', 60,
                #             datasets_meta_features[dataset_name], algorithms_meta_features, self.evaluate)
                state = self.env.reset()
                self.time_budget = int(datasets_meta_features[dataset_name]['time_budget'])
                ep_reward = 0.0         # Reward for this episode
                done = False          # Whether the game is finished
                loss = 0.0
                step = 0
                while not done:
                    step += 1
                    with torch.no_grad():
                        # Get and execute the next action for the current state
                        action = self.suggest(state)  # Sample action from policy
                        print("action = ", action)
                        A_star, A, delta_t = action
                        print("A = ", A)
                        print("delta_t = ", delta_t)
                        print("A_star = ", A_star)
                    next_state, reward, done = self.env.reveal((A_star, A, delta_t), validation_learning_curves[dataset_name], test_learning_curves[dataset_name]) # Step
                    ep_reward = ep_reward + reward.item()
                    if not self.evaluate:
                    # Save what the agent just learnt to the experience buffer.
                        self.memory.add(state, torch.tensor(A), reward, next_state, done)
                    loss = self.train()

                    state = next_state.clone().to(self.device, dtype=torch.float)
                    print("next_state = ", next_state)
                    print("reward = ", reward)
                    print("done = ", done)

                    print(f'Agent: {self.agent_name} . loss: {loss} . 'f'Reward: {ep_reward}')
                if self.epsilon != None and self.epsilon_decay_rate != None and not self.evaluate:
                    # self.epsilon = self.epsilon * self.epsilon_decay_rate
                    self.epsilon = max(self.epsilon * self.epsilon_decay_rate, 0.2)
                print("self.epsilon = ", self.epsilon)
                print("Epsilon:", self.epsilon)
        self.evaluate = True

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class DDQN_Network(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim1, hidden_dim2):
        super(DDQN_Network, self).__init__()

        # Q1 architecture
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.linear1 = nn.Linear(num_actions, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.relu(self.conv1(state))
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Replay():
    """
    Memory for storing experience
    """
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        :param state: The state before taking the action
        :param action: action taken
        :param reward: Reward for taking that action
        :param next_state: The state that the agent enters after taking the action
        :param done: Whether the agent finishes the game
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
          Return a batch of samples from the experience buffer
          :param batch_size: The number of sample that you want to take
          :return the batch of samples but decomposed into lists of states, actions, rewards, next_states
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # Random samples
        samples = random.sample(self.memory, self.batch_size)
        for s in samples:
            state = s[0]
            action = s[1]
            reward = s[2]
            next_state = s[3]
            done = s[4]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones
