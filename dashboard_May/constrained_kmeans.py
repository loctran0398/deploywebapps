#
# Author: Stanislaw Adaszewski, 2015
#

import networkx as nx
import numpy as np
import time
from scipy.spatial import distance

def constrained_kmeans(data, demand, maxiter=None, fixedprec=1e9):
	data = np.array(data)
	
	min_ = np.min(data, axis = 0)
	max_ = np.max(data, axis = 0)
    
	np.random.seed(1)
	C = min_ + np.random.random((len(demand), data.shape[1])) * (max_ - min_)
	M = np.array([-1] * len(data), dtype=np.int)
	
	itercnt = 0
	while True:
		itercnt += 1
		
		# memberships
		g = nx.DiGraph()
		g.add_nodes_from(xrange(0, data.shape[0]), demand=-1) # points
		for i in xrange(0, len(C)):
			g.add_node(len(data) + i, demand=demand[i])
		
		# Calculating cost...
		cost = np.array([np.linalg.norm(np.tile(data.T, len(C)).T - np.tile(C, len(data)).reshape(len(C) * len(data), C.shape[1]), axis=1)])
		# Preparing data_to_C_edges...
		data_to_C_edges = np.concatenate((np.tile([xrange(0, data.shape[0])], len(C)).T, np.tile(np.array([xrange(data.shape[0], data.shape[0] + C.shape[0])]).T, len(data)).reshape(len(C) * len(data), 1), cost.T * fixedprec), axis=1).astype(np.uint64)
		# Adding to graph
		g.add_weighted_edges_from(data_to_C_edges)
		

		a = len(data) + len(C)
		g.add_node(a, demand=len(data)-np.sum(demand))
		C_to_a_edges = np.concatenate((np.array([xrange(len(data), len(data) + len(C))]).T, np.tile([[a]], len(C)).T), axis=1)
		g.add_edges_from(C_to_a_edges)
		
		
		# Calculating min cost flow...
		f = nx.min_cost_flow(g)
		
		# assign
		M_new = np.ones(len(data), dtype=np.int) * -1
		for i in xrange(len(data)):
			p = sorted(f[i].iteritems(), key=lambda x: x[1])[-1][0]
			M_new[i] = p - len(data)
			
		# stop condition
		if np.all(M_new == M):
			# Stop
			return (C, M, f)
			
		M = M_new
			
		# compute new centers
		for i in xrange(len(C)):
			C[i, :] = np.mean(data[M==i, :], axis=0)
			
		if maxiter is not None and itercnt >= maxiter:
			# Max iterations reached
			return (C, M, f)

def compute_bic(C, M, nb_cluster, data_arr):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [C]
    labels  = M
    #number of clusters
    m = nb_cluster
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = data_arr.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(data_arr[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
    
    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)



#def main():
#	data = np.random.random((100, 8))
#	t = time.time()
#	(C, M, f) = constrained_kmeans(data_arr, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
#	print 'Elapsed:', (time.time() - t) * 1000, 'ms'
#	print 'C:', C
#	print 'M:', M
#	
#
#if __name__ == '__main__':
#	main()
