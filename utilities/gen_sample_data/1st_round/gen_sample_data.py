import copy
import numpy as np
from collections import defaultdict
import random
import math
import os
import shutil
from faker import Faker
Faker.seed(208)
random.seed(208)

a_path = 'a.txt'
b_path = 'b.txt'
c_path = 'c.txt'

class SigmoidCurve():
    def __init__(self, algo_name, a, b, c):
        self.algo_name = algo_name
        self.a = a
        self.b = b
        self.c = c
    def get_value(self, time_step):
        return min(1.0, max(0.0, self.a / (1 + math.exp(-self.b * (time_step - self.c)))))

if __name__ == "__main__":
    list_all_learning_curves = defaultdict(dict)
    root_dir = os.getcwd()
    sample_data_dir = os.path.join(root_dir, 'sample_data')
    meta_features_dir = os.path.join(sample_data_dir, 'dataset_meta_features')
    validation_dir = os.path.join(sample_data_dir, 'validation')
    test_dir = os.path.join(sample_data_dir, 'test')

    if not os.path.exists(sample_data_dir):
        os.makedirs(sample_data_dir)
    else:
        shutil.rmtree(sample_data_dir)
        os.makedirs(sample_data_dir)

    os.makedirs(meta_features_dir)
    os.makedirs(validation_dir)
    os.makedirs(test_dir)

    with open(a_path) as f:
        a = [[float(digit) for digit in line.split()] for line in f]
    with open(b_path) as f:
        b = [[float(digit) for digit in line.split()] for line in f]
    with open(c_path) as f:
        c = [[float(digit) for digit in line.split()] for line in f]

    number_of_datasets = 100
    number_of_algorithms = 20

    for i in range(number_of_datasets):
        for j in range(number_of_algorithms):
            curve = SigmoidCurve(j, a[i][j], b[i][j], c[i][j])
            list_all_learning_curves[i][j] = copy.deepcopy(curve)
            os.makedirs(validation_dir + '/' + str(i) + '/' + str(j))
            os.makedirs(test_dir + '/' + str(i) + '/' + str(j))

    #=== meta_features
    fake = Faker()
    names = [fake.unique.first_name() for i in range(number_of_datasets)]
    # print(names)
    meta_features = {}
    for i in range(number_of_datasets):
        dict = {'usage' : '\'AutoML challenge 2014\'',
                'name' : '\'' + names[i] + '\'',
                'task' : random.choice(['\'regression\'', '\'binary.classification\'',
                                        '\'multiclass.classification\'', '\'multilabel.classification\'']),
                'target_type' : random.choice(['\'Binary\'', '\'Categorical\'',
                                        '\'Numerical\'']),
                'feat_type' : random.choice(['\'Binary\'', '\'Categorical\'',
                                        '\'Numerical\'', '\'Mixed\'']),
                'metric' : random.choice(['\'bac_metric\'', '\'auc_metric\'',
                                        '\'f1_metric\'', '\'pac_metric\'',
                                        '\'a_metric\'', '\'r2_metric\'']),
                'time_budget' : random.randrange(100,1500,100),
                'feat_num' : str(random.randint(1,100)),
                'target_num' : str(random.randint(1,10)),
                'label_num' : str(random.randint(1,10)),
                'train_num' : str(random.randint(1,100000)),
                'valid_num' : str(random.randint(1,100000)),
                'test_num' : str(random.randint(1,100000)),
                'has_categorical' : str(random.randint(0,1)),
                'has_missing' : str(random.randint(0,1)),
                'is_sparse' : str(random.randint(0,1)),
                }
        meta_features[i] = dict

        with open(meta_features_dir + '/' + str(i) + '.info', 'w') as f:
            for key in dict:
                f.write(key + ' = ' + str(dict[key]) + "\n")
    print(meta_features)

    #=== learning_curves
    for i in range(number_of_datasets):
        for j in range(number_of_algorithms):
            lc = list_all_learning_curves[i][j]
            number_of_points = random.randint(5,10)
            timestamps = random.sample(range(0, 500), number_of_points)
            timestamps.sort()
            validation_scores = [lc.get_value(i) for i in timestamps]
            test_scores = [min(1.0, max(0.0, i + random.uniform(-0.05, -0.1))) for i in validation_scores]
            #
            print('\n#=== Learning curve of algorithm ' + str(j) + ' on dataset ' + str(i))
            print('timestamps = ', timestamps)
            print('validation_scores = ', validation_scores)
            print('test_scores = ', test_scores)

            with open(validation_dir + '/' + str(i) + '/' + str(j) + '/' + 'scores.txt', 'w') as f:
                f.write('times: ' + str(timestamps) + "\n")
                f.write('scores: ' + str(validation_scores) + "\n")

            with open(test_dir + '/' + str(i) + '/' + str(j) + '/' + 'scores.txt', 'w') as f:
                f.write('times: ' + str(timestamps) + "\n")
                f.write('scores: ' + str(test_scores) + "\n")
