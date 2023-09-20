import random
import numpy as np
import torch
import pdb

class Agent():
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        ### TO BE IMPLEMENTED ###
        self.nA = number_of_algorithms

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
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
        self.models_to_be_run = [i for i in range(self.nA)]
        # self.s = torch.ones(2, 1, self.nA)
        self.s = torch.ones(3, 1, self.nA)
        self.s[0] = (-1) * self.s[0] # Current validation performance: R_validation_C_A
        self.s[1] = 0.0 * self.s[1] # Time spent: C_A
        self.s[2] = (100) * self.s[0] # 3nd element along the 1st dimension of s: family of the chosen algo.
        self.total_time_budget = int(dataset_meta_features['time_budget'])
        self.remaining_time = int(dataset_meta_features['time_budget'])
        self.algorithms_meta_features = algorithms_meta_features

    def meta_train(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):
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
        pass

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
        if observation==None:
            A_star = None
        else:
            A, C_A, R_validation_C_A = observation
            # print(A, C_A, R_validation_C_A)
            self.s[0,0,A] = R_validation_C_A
            self.s[1,0,A] = C_A
            self.s[2,0,A] = self.get_algo_family(A)
            A_star = self.get_A_star()

        if len(self.models_to_be_run)==0:
   
            # next_A = self.get_A_star_random()
            next_A = self.get_A_star_epsilon_greedy()
            # it's possible that some actions have time spent = 0
            if self.s[1,0,next_A] == 0:
                # in this case, we increment the time by self.total_time_budget/50
                delta_t = self.total_time_budget/50
            else:
                # otherwise double the time spent
                delta_t = float(min(self.s[1,0,next_A]*2, self.remaining_time))
        else:
            next_A = self.models_to_be_run.pop(0)
            delta_t = self.total_time_budget/50

        self.remaining_time -= delta_t

        action = (A_star, next_A, delta_t)

        return action

    def get_A_star(self):
        return torch.argmax(self.s[0]).item()

    def get_A_star_random(self):
        best_algo = []
        # get top algo for each family
        for fam_i in [0,1]:
            best_algo.append(torch.argmax(self.s[0][(self.s[2] == fam_i)]).item() + fam_i *10)

        # get random A from best_algo
        A_star = random.choice(best_algo)
        return A_star
        # return torch.argmax(self.s[0]).item()

    def get_A_star_epsilon_greedy(self, epsilon=0.1):
        # get top algo for each family
        best_algo = []
        for fam_i in [0,1,]:
            best_algo.append(torch.argmax(self.s[0][(self.s[2] == fam_i)]).item() + fam_i *10)
        # epsilon-greedy
        p = np.random.random()
        if p<epsilon:
            A_star = random.choice(best_algo)
        else:
            A_star = torch.argmax(self.s[0]).item()
        print('best_algo: ', best_algo)
        print('best_algo R_val: ', [self.s[0][0][i] for i in best_algo])
        print('epsilon-greedy p:', p)
        print('A_star:', A_star)
        return A_star


 


    def get_algo_family(self, A):
        return int(self.algorithms_meta_features[str(A)]['meta_feature_0'])

