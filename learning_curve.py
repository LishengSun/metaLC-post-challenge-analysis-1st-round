import numpy as np
import json

class Learning_Curve():
    """
    MODEL A LEARNING CURVE OF AN ALGORITHM ON A DATASET
    """
    def __init__(self, file_path):
        """
        Initialize a learning curve

        :param file_path: path to obtain the learning curve
        """
        self.file_path = file_path
        self.scores, self.timestamps = self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load timestamps and scores from a given path to build a learning curve

        :param file_path: path to obtain the learning curve
        :return scores: list of performance scores obtained by an algorithm on a dataset
        :return timestamps: list of timestamps when the scores are made
        """
        scores, timestamps = [], []
        try:
            with open(file_path, "r") as data:
                lines = data.readlines()
                dictionary = {line.split(":")[0]:line.split(":")[1] for line in lines}
                timestamps = np.around(json.loads(dictionary['times']), decimals=2)
                scores = np.around(json.loads(dictionary['scores']), decimals=2)

        # If the data is missing, set timestamp = 0 and score = 0 as default
        except FileNotFoundError:
            scores.append(0.0)
            timestamps.append(0.0)
            dataset_name = file_path.split('/')[7]
            algo_name = file_path.split('/')[8]
            print("*Warning* Learning curve of algorithm \"{}\" on dataset \"{}\" is missing, replaced by 0 as default!".format(algo_name, dataset_name))
        return scores, timestamps

    def get_last_point_within_delta_t(self, delta_t, C_A):
        """
        Return the last achievable point on the learning curve given the allocated time budget delta_t

        :param delta_t: allocated time budget given by the agent
        :param C_A: the timestamp of the last point on the learning curve (x-coordinate of current position on the learning curve)
        :return
        """
        temp_time = C_A + delta_t

        for i in range(len(self.timestamps)):
            if temp_time<self.timestamps[i]:
                if i==0: #if delta_t is not enough to get the first point, the agent wasted it for nothing!
                    score, timestamp = 0.0, 0.0
                else: # return the last achievable point
                    score, timestamp = self.scores[i-1], self.timestamps[i-1]
                return score, timestamp

        # If the last point on the learning curve is already reached, return it
        return self.scores[-1], self.timestamps[-1]
