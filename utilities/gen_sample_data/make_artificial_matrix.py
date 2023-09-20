import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from collections import defaultdict
import copy
import math
from matplotlib.pyplot import figure
import random
from matplotlib.cm import get_cmap
name = "tab20"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

random.seed(208)

class SigmoidCurve():
	def __init__(self, algo_name, a, b, c):
		self.algo_name = algo_name
		self.a = a
		self.b = b
		self.c = c

	def get_value(self, time_step):
		return self.a / (1 + math.exp(-self.b * (time_step - self.c)))

def gen_abc(num_rows, num_cols, rank):
	_, a = make_SVD_artificial_matrix(num_rows,num_cols,rank, reverse=False)
	a = minmax_scale(a, feature_range=(0.5, 0.95), axis=0)
	with open("a.txt", "w") as f:
		np.savetxt(f, a)


	_, b = make_SVD_artificial_matrix(num_rows,num_cols,rank, reverse=True)
	b = minmax_scale(b, feature_range=(-2, 5), axis=0)
	with open("b.txt", "w") as f:
		np.savetxt(f, b)

	_, c = make_SVD_artificial_matrix(num_rows,num_cols,rank, reverse=True)
	c = minmax_scale(c, feature_range=(-1, 1), axis=0)
	with open("c.txt", "w") as f:
		np.savetxt(f, c)

def plot_abc():
	with open('a.txt') as f:
		a = [[float(digit) for digit in line.split()] for line in f]
		g = sns.clustermap(data=a, yticklabels=0, xticklabels=0, cmap="Blues")
		ax = g.ax_heatmap
		ax.set_ylabel("Dataset")
		ax.set_xlabel("Algorithm")
		g.fig.set_size_inches(6,6)
		g.savefig("a_clustermap_" + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')

	with open('b.txt') as f:
		b = [[float(digit) for digit in line.split()] for line in f]
		g = sns.clustermap(data=b, yticklabels=0, xticklabels=0, cmap="Blues")
		ax = g.ax_heatmap
		ax.set_ylabel("Dataset")
		ax.set_xlabel("Algorithm")
		g.fig.set_size_inches(6,6)
		g.savefig("b_clustermap_" + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')

	with open('c.txt') as f:
		c = [[float(digit) for digit in line.split()] for line in f]
		g = sns.clustermap(data=c, yticklabels=0, xticklabels=0, cmap="Blues")
		ax = g.ax_heatmap
		ax.set_ylabel("Dataset")
		ax.set_xlabel("Algorithm")
		g.fig.set_size_inches(6,6)
		g.savefig("c_clustermap_" + ".svg", dpi=1200, bbox_inches='tight', pad_inches = 0, format='svg')

def make_SVD_artificial_matrix(num_row, num_col, rank, reverse):
	"""
	Make a matrix with shape (num_row, num_col) and rank = rank
	M = U S V
	where S is diagonal with values decreasing exponentially
	"""
	if rank > min(num_row, num_col):
		raise ValueError('Rank not realizable.')

	from scipy.stats import ortho_group
	U = ortho_group.rvs(dim=num_row, random_state=208)
	V = ortho_group.rvs(dim=num_col, random_state=208)
	S = np.zeros((num_row, num_col))
	if reverse:
		for d in range(rank): # make exponential decay
			S[d,d] = 100*np.exp(d)
	else:
		for d in range(rank): # make exponential decay
			S[d,d] = 100*np.exp(-d)
	M = np.dot(U, np.dot(S, V))
	return S, M


if __name__ == '__main__':
	num_rows = 100
	num_cols = 40
	rank = 20
	times = [(i)/10 for i in range(1, 11)]
	# palette = sns.color_palette("Blues")
	gen_abc(num_rows, num_cols, rank)
	plot_abc()

	with open('a.txt') as f:
		a = [[float(digit) for digit in line.split()] for line in f]
	with open('b.txt') as f:
		b = [[float(digit) for digit in line.split()] for line in f]
	with open('c.txt') as f:
		c = [[float(digit) for digit in line.split()] for line in f]

	list_all_learning_curves = defaultdict(dict)

	for i in range(num_rows):
		for j in range(num_cols):
			curve = SigmoidCurve(str(j), a[i][j], b[i][j], c[i][j])
			list_all_learning_curves[str(i)][str(j)] = copy.deepcopy(curve)


	plt.figure(figsize=(10,6))
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_prop_cycle(color=colors)
	for j in range(num_cols):
		y = [list_all_learning_curves['0'][str(j)].get_value(k) for k in times]
		print(y)
		# plt.plot(times, y, label = "Algorithm " + str(j))
		ax.step(times, y, where='post')
		# ax.scatter(times, y, s=80, marker="o")

	plt.xlabel('training size')
	plt.xticks([(i)/10 for i in range(1, 11)])
	# plt.ylim(bottom=0.0)
	plt.ylabel('performance metric (e.g. test accuracy)')
	plt.title('Learning curve samples of algorithms on one dataset from the Artificial Meta-dataset')
	# plt.legend(loc=4)
	# plt.show()
	# figure(figsize=(8, 6))

	plt.tight_layout()
	plt.savefig("artificial_samples" + ".svg", dpi=1200, format='svg', bbox_inches='tight')

	# S_diago = np.diag(S)
	# print("#### S ####", S)
	# print("#### M ####", M)
	# print("#### S_diago ####", S_diago)
	# print(np.min(M))
	# print(np.max(M))
	# x = range(len(S_diago))
	# plt.bar(x, S_diago, label='Diag(d) = 100 * exp(-d)')

	# plt.ylabel('Diagonal values of S')
	# plt.xlabel('d')
	# plt.title('artificial matrix (50*20) M = USV')
	# plt.legend()
	# # plt.savefig(os.path.join(os.getcwd(),'./artificial/r50c20r20_S_diagonal'))
	# plt.show()
	# np.savetxt(os.path.join(os.getcwd(),'./artificial/r50c20r20.data'), M)
