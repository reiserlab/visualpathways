""" 
adj matrix normalized by row = transtion matrix
adj matrix normalized by column = better as weight propagation
transpose = inverse propagation
inverse with constanst input at the sink ~ page rank
forward with constant input at the source ~ steady state
use the convention that subsequent props as right multiplications

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
# import plotly.io as pio
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

m_adj = np.array([[0,0,0,0,0],
                [1/2,0,0,0,0],
                [1/2,0,0,0,0],
                [0,1/2,0,0,0],
                [0,1/2,1,0,0]])

#    1
#  /   \
# 3 --> 2

m_adj = np.array([[0,0,0],
                [1/2,0,0],
                [1/2,1,0]])

# normalize by column
m_adj = m_adj / m_adj.sum(axis=0)
m_adj = np.nan_to_num(m_adj)

# set diag to 1 for columns that sum to zero 
m_adj_loop = m_adj.copy()
for i in range(m_adj_loop.shape[1]):
    if m_adj_loop[:,i].sum() == 0:
        m_adj_loop[i,i] = 1

m_trans = m_adj.T
# normalize by row
m_trans = m_trans / m_trans.sum(axis=1)[:, np.newaxis]
m_trans = np.nan_to_num(m_trans)

m_trans_loop = m_adj_loop.T

# NB 
# np.linalg.matrix_power(m_trans_loop, inf) -> steady state probability with single input
# prop_series_inf(m_trans) -> steady state population with constant input

# m = np.array([[0,0.5,0.5,0,0,0],
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
    for i in range(1, n+1):
        prop_series += np.linalg.matrix_power(arr, i)

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
