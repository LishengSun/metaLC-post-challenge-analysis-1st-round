# import numpy as np
# import os
# root = ''
# num = 0
#
# def format_float(num):
#     return np.format_float_positional(num, trim='-')
#
# for i in range(4):
#     mf_0 = i
#     for j in range(2):
#         mf_1 = j
#         for k in range(5):
#             mf_2 = 0.1/(10**k)
#             print("mf_0=", mf_0)
#             print("mf_1=", mf_1)
#             print("mf_2=", format_float(mf_2))
#
#             with open(os.getcwd() + '/' + 'sample_data/algorithms_meta_features/' + str(num) + '.info', 'w') as f:
#                 f.write('meta_feature_0 = ' + str(mf_0) + "\n")
#                 f.write('meta_feature_1 = ' + str(mf_1) + "\n")
#                 f.write('meta_feature_2 = ' + str(format_float(mf_2)) + "\n")
#
#             num += 1

import matplotlib
import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import figure
plt.rcParams['font.size'] = '20'
fig, ax = plt.subplots(figsize=(8, 6))


p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scores = [0.5, 0.55, 0.6, 0.55, 0.6, 0.6, 0.65, 0.7, 0.7, 0.65]
ax.step(p, scores, 'g', where='post')
ax.scatter(p, scores, s=80, marker="o", label = 'aa')
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xlabel('fraction of the training set')
plt.ylabel('training/validation/test score')
plt.show()
