import numpy as np
import igraph
import leidenalg as la
from sklearn.utils.validation import check_random_state
    

def find_la_modules(A, gamma=1, seed=None):
    # Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
    g = igraph.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter = gamma, weights=A[A>0], seed=seed)
    
    return part.membership, part.quality()/A.sum().sum()


def find_infomap(A, trials=10):
    # Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
    g = igraph.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    
    part = g.community_infomap(edge_weights=A[A.nonzero()], trials=trials)
    
    return part.membership, part.codelength


def find_consensus(assignments, null_func=np.mean, gamma=1, return_agreement=False, seed=None, model='Modularity'):
    """
    Modified from https://netneurotools.readthedocs.io/en/latest/_modules/netneurotools/cluster.html#find_consensus
    """
    
    rs = check_random_state(seed)
    samp, comm = assignments.shape
    
    ### JH edit
    if comm==1:
        return np.squeeze(assignments).astype(int)+1, 0

    # create agreement matrix from input community assignments and convert to
    # probability matrix by dividing by `comm`
    agreement = find_agreement(assignments, buffsz=samp) / comm

    # generate null agreement matrix and use to create threshold
    null_assign = np.column_stack([rs.permutation(i) for i in assignments.T])
    null_agree = find_agreement(null_assign, buffsz=samp) / comm
    threshold = null_func(null_agree)

    # run consensus clustering on agreement matrix after thresholding
    consensus = find_consensus_und(agreement, threshold, gamma=gamma, reps=10, seed=seed, model=model)

    if return_agreement:
        return consensus.astype(int), agreement * (agreement > threshold)

    return consensus.astype(int)


def find_consensus_und(D, tau, gamma=1, reps=1000, seed=None, model='Modularity'):
    '''
    Modified from https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/clustering.py
    '''
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology

        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci

        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
            # count+=1
            # print count,c,dup
            # if count>10:
            #	class QualitativeError(): pass
            #	raise QualitativeError()
        return np.transpose(ciu)

    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                if model=='Modularity':
                    cis[:, i], _ = find_la_modules(dt, gamma=gamma, seed=seed)
                elif model=='Infomap':
                    cis[:, i] = find_infomap(dt, seed=seed)
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = find_agreement(cis) / reps

    return np.squeeze(ciu + 1)

def find_agreement(ci, buffsz=150):
    '''
    Copied from https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/clustering.py
    '''
    ci = np.array(ci)
    n_nodes, n_partitions = ci.shape

    if n_partitions <= buffsz: # Case 1: Use all partitions at once
        ind = dummyvar(ci)
        D = np.dot(ind, ind.T)
    else: # Case 2: Add together results from subsets of partitions
        a = np.arange(0, n_partitions, buffsz)
        b = np.arange(buffsz, n_partitions, buffsz)
        if len(a) != len(b):
            b = np.append(b, n_partitions)
        D = np.zeros((n_nodes, n_nodes))
        for i, j in zip(a, b):
            y = ci[:, i:j]
            ind = dummyvar(y)
            D += np.dot(ind, ind.T)

    np.fill_diagonal(D, 0)
    return D

def dummyvar(cis, return_sparse=False):
    '''
    Copied from https://github.com/aestrivex/bctpy/blob/32c7fe7345b281c2d4e184f5379c425c36f3bbc7/bct/utils/miscellaneous_utilities.py
    '''
    # num_rows is not affected by partition indexes
    n = np.size(cis, axis=0)
    m = np.size(cis, axis=1)
    r = np.sum((np.max(len(np.unique(cis[:, i])))) for i in range(m))
    nnz = np.prod(cis.shape)

    ix = np.argsort(cis, axis=0)
    # s_cis=np.sort(cis,axis=0)
    # FIXME use the sorted indices to sort by row efficiently
    s_cis = cis[ix][:, range(m), range(m)]

    mask = np.hstack((((True,),) * m, (s_cis[:-1, :] != s_cis[1:, :]).T))
    indptr, = np.where(mask.flat)
    indptr = np.append(indptr, nnz)

    import scipy.sparse as sp
    dv = sp.csc_matrix((np.repeat((1,), nnz), ix.T.flat, indptr), shape=(n, r))
    return dv.toarray()