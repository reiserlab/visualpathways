""" 
Use the convention that subsequent props as right multiplications, initial state is a row vector on the left.

adj matrix normalized, with row indices as input/upstream nodes, columns indices as output/downstream nodes.
adj matrix normalized by column => better interpreted as forward propagation for activity.

Backward propagation interpreted as distributing contribution,
which is formalized as either multipling an initial state column vector on the right, or transposing everything.

Note, transpose is now row-normalized, which is a transition matrix.

Infinite series of backward propagation with constant input at the sink ~ pagerank contribution
Infinite series of forward propagation with constant input at the source ~ steady state activity

"""

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
# load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
# folder_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# %% Import libraries
import pandas as pd
import numpy as np
# import re

# %%
# define adj matrix, upward directed
#     1
#    / \
#   3   2
#    \ /  \
#     5    4

# m_adj = np.array([[0,0,0,0,0],
#                 [1/2,0,0,0,0],
#                 [1/2,0,0,0,0],
#                 [0,1/2,0,0,0],
#                 [0,1/2,1,0,0]])
# # normalize by column
# m_adj = m_adj / m_adj.sum(axis=0)
# m_adj = np.nan_to_num(m_adj)

# m_trans = m_adj.T
# # normalize by row if not already
# m_trans = m_trans / m_trans.sum(axis=1)[:, np.newaxis]
# m_trans = np.nan_to_num(m_trans)


#  m = np.array([[0,0.5,0.5,0,0,0],
#                 [0,0,0.5,0.5,0,0],
#                 [0,0,0,0,0,0],
#                 [0,0,0,0,0.25,0.25],
#                 [0,0,0,0,0,0],
#                 [0,0,0,0,0,0]])


# %%
# a function to compute the series of powers of adj matrix, m + m^2 + m^3 + ... + m^n
def prop_series(m, n):
    if isinstance(m, pd.DataFrame):
        arr = m.values
    else:
        arr = m
        
    prop_series = np.zeros(arr.shape)
    for i in range(n):
        if i == 0:
            prop_series += arr
        else:
            arr = arr @ arr
            prop_series += arr

    # convert to df if the input is df
    if isinstance(m, pd.DataFrame):
        prop_series = pd.DataFrame(prop_series, index=m.index, columns=m.columns)

    return prop_series

# for infinite series, Neumann series
def prop_series_inf(m):
    if isinstance(m, pd.DataFrame):
        arr = m.values
    else:
        arr = m
    
    # if np.max(m) == 1:
    #     m = m/2 #convergence

    # identity matrix minus m, then inverse
    idmm = np.identity(arr.shape[0]) - arr

    # check if idmm is invertable
    if np.linalg.det(idmm) == 0:
        print("Matrix is singular, use pseudoinverse")
        # # compute SVD
        # u, s, vh = np.linalg.svd(idmm)
        # # get 1/s, replace inf with zero
        # s_inv = np.where(s == 0, 0, 1/s)
        # # pseudo inverse
        # prop_series = vh.T @ np.diag(s_inv) @ u.T

        prop_series = np.linalg.pinv(idmm)
    else:
        prop_series = np.linalg.inv(idmm)

    # remove identity matrix
    prop_series = prop_series - np.identity(arr.shape[0])

    # prop_series = np.linalg.inv(idmm)

    # convert to df if the input is df
    if isinstance(m, pd.DataFrame):
        prop_series = pd.DataFrame(prop_series, index=m.index, columns=m.columns)

    return prop_series