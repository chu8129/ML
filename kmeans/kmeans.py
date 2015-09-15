from numpy import *
def load_data_set(filename):
	data_mat=[]
	with open(filename,'r') as fr:
		for line in fr.readlines():
			currline=line.strip().split('\t')
			data_mat.append(map(float,currline))
	return data_mat

def dist_eclud(veca,vecb):
	return sqrt(sum(power(veca-vecb,2)))

def rand_cent(data_set,k):
	n=shape(data_set)[1]
	centroids=mat(zeros((k,n)))
	for j in range(n):
		min_j=min(data_set[:,j])
		range_j=float(max(data_set[:,j])-min_j)
		centroids[:,j]=min_j+random.rand(k,1)*range_j
	return centroids

