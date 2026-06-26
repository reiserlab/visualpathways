from pathlib import Path
import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from utils.hex_hex import all_hex_df
from cmath import pi
import networkx as nx
from pyvis.network import Network

from utils.config import EYEMAP_DIR, HTML_FIG_DIR, SIDE
from utils.geometry import cart2sph, sph2Mollweide
from utils.ol_color import OL_COLOR

# === From main src/utils/plotting_functions.py (mollweide / polar / pyvis family) ===

def plot_mollweide_projection(
    data:pd.DataFrame
  , feature_col:str
  , cmin=-0.2
  , cmax=0.2
  , cmap='RdBu_r'
) -> go.Figure:
    """
    Generates a heatmap to visualize the value of column features per column using the 
    mollweide projection.

    Parameters
    ----------
    data : pd.DataFrame
        data frame containing the values of column features per column with (at least) columns
        `hex1_id` : int
            hex1 coordinates of column
        `hex2_id` : int
            hex2 coordinates of column
    feature_col : str
        column of 'data' under investigation

    Returns
    -------
    fig : go.Figure
        Heatmap
    """
    #find hex to xyz mapping
    ucl_hex = pd.read_pickle(EYEMAP_DIR / 'mcns_20240701' / f'ucl_hex_{SIDE}.pkl')
    rtp2 = cart2sph(ucl_hex[['x','y','z']].values)
    xy = sph2Mollweide(rtp2[:,1:3])
    xy[:,0] = -xy[:,0] # flip x axis
    xypq_moll = np.concatenate((xy, ucl_hex[['p','q']].values), axis=1)
    # convert to df and change type of the last 2 columns to int
    xypq_moll = pd.DataFrame(xypq_moll, columns=['x','y','p','q'])
    xypq_moll[['p','q']] = xypq_moll[['p','q']].astype(int)

    data = data.merge(xypq_moll, left_on=['hex1_id','hex2_id'], right_on=['q','p'], how='left')
    fig, ax = plt_mollweide()
    ax.scatter(data['x'].values, data['y'].values, c=data[feature_col].values, 
                cmap=cmap, vmin=cmin, vmax=cmax, )
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    cbar.mappable.set_clim(cmin, cmax)    
    cbar.set_label(feature_col)
    fig.set_size_inches(16,8)

    return fig


def generate_random_color():
    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    # Format RGB values as hexadecimal string
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    return color_hex


def plot_1d_polar(
    data:pd.DataFrame
  , theta_col:str
  , rad_col:str
  , color:str='#1f77b4'
) -> go.Figure:
    """
    Plot 1d polar distribution

    Parameters
    ----------
    data : pd.DataFrame
        data frame with at least two columns
    x_col : str
        name of column in data to plot on x-axis
    y_col : str
        name of column in data to plot on y-axis
    color : str
        color of the line and marker

    Returns
    -------
    fig : go.Figure
        lineplot
    """
    #plotting parameters
    h = 600
    w = 600    
    
    fig = go.Figure(data=go.Scatterpolar(
                                    theta=data[theta_col]\
                                  , r=data[rad_col]\
                                  , mode='lines+markers'\
                                  , line={'color': color, 'width': 3}
                                  , marker={'color': color, 'size': 10}
                                  , opacity=.6
                                  , showlegend=False
    ))
    fig.update_layout(
        font={
            "family":'Arial'
          , "size" : 14
        }
      , height=h
      , width=w
      , polar=dict(
            radialaxis=dict(
                visible=True
              , gridcolor='black'
              , tickvals=np.linspace(0,1,11)
              , range=[0, 1.2*data[rad_col]]
            )
          , angularaxis=dict(
                visible=True
              , gridcolor='black'
            )
          , bgcolor='white'
        )
    )
    fig.update_xaxes(
        showline=True
      , linewidth=1
      , linecolor='black'
      , mirror=True
    )
    fig.update_yaxes(
        showline=True
      , linewidth=1
      , linecolor='black'
      , mirror=True
    )
    return fig


def plot_1d_distr(
    data:pd.DataFrame
  , x_col:str
  , y_col:str
  , color:str='#1f77b4'
) -> go.Figure:
    """
    Plot 1d distribution

    Parameters
    ----------
    data : pd.DataFrame
        data frame with at least two columns
    x_col : str
        name of column in data to plot on x-axis
    y_col : str
        name of column in data to plot on y-axis
    color : str
        color of the line and marker

    Returns
    -------
    fig : go.Figure
        lineplot
    """
    #plotting parameters
    h = 400
    w = 800    
    
    fig = go.Figure(data=go.Scatter(
                                    x=data[x_col]\
                                  , y=data[y_col]\
                                  , mode='lines+markers'\
                                  , line={'color': color, 'width': 3}
                                  , marker={'color': color, 'size': 10}
                                  , opacity=.6
                                  , showlegend=False
    ))
    fig.update_layout(
        font={
            "family":'Arial'
          , "size" : 14
        }
      , xaxis_title=x_col
      , yaxis_title=y_col
      , xaxis_range=[0, 1.2*data[x_col].max()]
      , yaxis_range=[0, 1.2*data[y_col].max()]
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
      , height=h
      , width=w
    )
    fig.update_xaxes(
        showline=True
      , linewidth=1
      , linecolor='black'
      , mirror=True
    )
    fig.update_yaxes(
        showline=True
      , linewidth=1
      , linecolor='black'
      , mirror=True
    )
    return fig


def plot_col_heatmap(
    data:pd.DataFrame
  , feature_col:str
  , cmin=-0.2
  , cmax=0.2
  , cmap='RdBu_r'
) -> go.Figure:
    """
    Generates a heatmap to visualize the value of column features per column using the 
    hex_id coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        data frame containing the values of column features per column with (at least) columns
        `hex1_id` : int
            hex1 coordinates of column
        `hex2_id` : int
            hex2 coordinates of column
    feature_col : str
        column of 'data' under investigation

    Returns
    -------
    fig : go.Figure
        Heatmap
    """
    #plotting parameters
    # pio.kaleido.scope.mathjax = None
    w = 625+50
    h = 660
    dotsize = 15
    symbol_number = 15
    column_df = all_hex_df()
    hex1_vals_empty = column_df['hex1_id'].astype('float').values
    hex2_vals_empty = column_df['hex2_id'].astype('float').values
    x_vals_empty = hex2_vals_empty.reshape((-1,1)).flatten()-hex1_vals_empty.reshape((-1,1)).flatten()
    y_vals_empty = hex2_vals_empty.reshape((-1,1)).flatten()+hex1_vals_empty.reshape((-1,1)).flatten()
    mul_fac = 1
    tot_max = np.multiply([column_df['hex1_id'].max() + column_df['hex2_id'].max()],  mul_fac)
    tot_min = np.multiply([column_df['hex1_id'].min() - column_df['hex2_id'].max()],  mul_fac)

    #what to plot
    x_vals = data['hex2_id'].values - data['hex1_id'].values
    y_vals = data['hex2_id'].values + data['hex1_id'].values
    col_vals = data[feature_col].values
    customdata= np.stack((data['hex1_id'], data['hex2_id'], data[feature_col]), axis=-1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals_empty
      , y=y_vals_empty
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'white'
          , 'line': {'width': 0.5, 'color': 'black'}
        }
      , showlegend=False))
    fig.add_trace(go.Scatter(
        x=x_vals
      , y=y_vals
      , mode='markers+text'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': col_vals
          , 'line': {'width': 0.5, 'color': 'lightgrey'}
          , 'colorscale' : cmap
          , 'cmin': cmin
          , 'cmax': cmax
          , 'opacity': 1
          , 'colorbar' : {'title': feature_col, 'thickness': 10}
      }
      , customdata=customdata
      , hovertemplate = 'hex1_id: %{customdata[0]}<br>hex2_id: %{customdata[1]}<br>value: %{customdata[2]}'
      , showlegend=False))
        
    # Update the layout of the figure
    fig.update_layout(
        font={
            "family":'Arial'
          , "size" : 14
        }
      , yaxis_range=[tot_min , tot_max + tot_max/10]
      , xaxis_range=[tot_min, tot_max + tot_max/10]
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
      , height=h
      , width=w
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, title_standoff=0)
    fig.update_yaxes(showgrid=False, showticklabels=False, title_standoff=2)
    
    return fig


def plot_rel_col_heatmap(
    data:pd.DataFrame
  , feature_col:str
  , hex_range:str
  , cmin=0.01
  , cmax=0.2
) -> go.Figure:
    """
    Generates a heatmap to visualize the value of column features per relative column using the 
    hex_id coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        data frame containing the values of column features per column with (at least) columns
        `hex1_id` : int
            hex1 coordinates of column
        `hex2_id` : int
            hex2 coordinates of column
    feature_col : str
        column of 'data' under investigation
    hex_range : str
        'small', 'medium or 'large

    Returns
    -------
    fig : go.Figure
        Heatmap
    """
    #plotting parameters
    # pio.kaleido.scope.mathjax = None
    cmap = 'Hot_r' #'Inferno_r'
    h = 550
    w = 580
    symbol_number = 15
    if hex_range=='large':
        hex_max = 30
        hex_min = -30 
        y_range = [-30, 30]
        x_range = [-20, 20]
        dotsize = 13.5
    elif hex_range=='small':
        hex_max = 6
        hex_min = -6 
        y_range = [-6, 6]
        x_range = [-4, 4]
        dotsize = 70
    elif hex_range=='medium':
        hex_max = 12
        hex_min = -12
        y_range = [-12, 12]
        x_range = [-8, 8]
        dotsize = 35
    hex1_vals_empty, hex2_vals_empty = np.meshgrid(np.array(range(hex_min, hex_max+1)), np.array(range(hex_min, hex_max+1)))
    x_vals_empty = hex2_vals_empty.reshape((-1,1)).flatten()-hex1_vals_empty.reshape((-1,1)).flatten()
    y_vals_empty = hex2_vals_empty.reshape((-1,1)).flatten()+hex1_vals_empty.reshape((-1,1)).flatten()
    
    #what to plot
    x_vals = data['hex2_id'].values - data['hex1_id'].values
    y_vals = data['hex2_id'].values + data['hex1_id'].values
    col_vals = data[feature_col].values
    # cmin = 0.01
    # cmax = 0.2 #np.quantile(col_vals,0.99)
    anno_txt = np.array(['%.2f'%c for c in col_vals])
    if hex_range!='large':
      anno_txt[col_vals<cmin] = ''
    else:
      anno_txt = ''
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals_empty
      , y=y_vals_empty
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'white'
          , 'line': {'width': 0.5, 'color': 'lightgrey'}
        }
      , showlegend=False))
    fig.add_trace(go.Scatter(
        x=x_vals
      , y=y_vals
      , mode='markers+text'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': col_vals
          , 'line': {'width': 0.5, 'color': 'lightgrey'}
          , 'colorscale' : cmap
          , 'cmin': cmin
          , 'cmax': cmax
        }
      , text = anno_txt
      , textfont=dict(color='white', family='Arial', size=18)
      , showlegend=False))
    fig.add_trace(go.Scatter(
        x=[0]
      , y=[0]
      , mode='markers'
      , marker_symbol=115
      , marker={
            'size': dotsize
          , 'color': 'black'
        }
      , showlegend=False))
        
    # Update the layout of the figure
    fig.update_layout(
        font={
            "family":'Arial'
          , "size" : 14
        }
      # , xaxis_title='   ==========> H'
      # , yaxis_title='   ==========> V'
      , yaxis_range=y_range
      , xaxis_range=x_range
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
      , height=h
      , width=w
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, title_standoff=0)
    fig.update_yaxes(showgrid=False, showticklabels=False, title_standoff=2)
    
    return fig


def plt_mollweide(
        fig_w=10.5,
        fig_h=4.5,
        col='k'
        ) -> tuple:
    """
    Plot Mollweide guidelines
    """

    # define guidelines
    ww = np.stack((np.linspace(0,180,19), np.repeat(-180,19)), axis=1)
    w = np.stack((np.linspace(180,0,19), np.repeat(-90,19)), axis=1)
    m = np.stack((np.linspace(0,180,19), np.repeat(0,19)), axis=1)
    e = np.stack((np.linspace(180,0,19), np.repeat(90,19)), axis=1)
    ee = np.stack((np.linspace(0,180,19), np.repeat(180,19)), axis=1)
    pts = np.vstack((ww,w,m,e,ee))
    rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
    meridians_xy = sph2Mollweide(rtp[:,1:3])

    pts = np.stack((np.repeat(45,37), np.linspace(-180,180,37)), axis=1)
    rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
    n45_xy = sph2Mollweide(rtp[:,1:3])
    pts = np.stack((np.repeat(90,37), np.linspace(-180,180,37)), axis=1)
    rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
    eq_xy = sph2Mollweide(rtp[:,1:3])
    pts = np.stack((np.repeat(135,37), np.linspace(-180,180,37)), axis=1)
    rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
    s45_xy = sph2Mollweide(rtp[:,1:3])

    # plot guidelines
    plt.rcParams["figure.figsize"] = [fig_w, fig_h]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(meridians_xy[:,0], meridians_xy[:,1], '-', color=col, linewidth=1.0)
    ax.plot(n45_xy[:,0], n45_xy[:,1], '-', color=col, linewidth=1)
    ax.plot(eq_xy[:,0], eq_xy[:,1], '-', color=col, linewidth=1)
    ax.plot(s45_xy[:,0], s45_xy[:,1], '-', color=col, linewidth=1)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("azimuth")
    ax.set_ylabel("elevation")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig, ax


def plt_mercator(
        fig_w=8.5,
        fig_h=4.5,
        col='k'
        ) -> tuple:
    """
    Plot Mercator guidelines
    """

    plt.rcParams["figure.figsize"] = [fig_w, fig_h]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_aspect('equal')

    # set axes ticks and labels
    x = np.arange(-180,180,45)
    xtick = x/180*np.pi
    y = np.arange(-75,75,25)
    ytick = np.log(np.tan(np.pi/4 + y/180*np.pi/2))

    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    ax.set_xlabel("azimuth")
    ax.set_ylabel("elevation")

    return fig, ax


def calc_node_sizes_Angel(
        w: np.ndarray
        ) -> np.ndarray:
    """
    Compute node sizes based on Angel's formula
    """
    w_log = np.log(w)/np.log(5)
    node_w_log_norm = (w_log + 7) / 8
    node_w_log_norm[node_w_log_norm < 0] = 0
    node_w_log_norm[node_w_log_norm > 1] = 1
    node_w_log_norm = node_w_log_norm + 0.1
    return node_w_log_norm


def plot_pyvis_Angel(m_adj, m_trans_inf, n_source=None, n_target=None, n_path=None, ids_layer=None, save_path=None, include_nonprimary_links=False):
    print('Setting up network in pyvis..')

    # Angel's significance as node size, backward transition
    v0 = [1 if np.isin(m_adj.index[i], n_target['bodyId'].values) else 0 for i in range(len(n_path))]
    node_size = calc_node_sizes_Angel(v0 @ m_trans_inf)

    # connection weights, forward
    edge_weights = 10*m_adj.to_numpy().flatten()
    edge_weights = edge_weights[edge_weights > 0]
    # edge_weights = 0.5 + 3*(2*np.median(edge_weights) - np.min(edge_weights) - edge_weights)/(2*np.median(edge_weights) - 2*np.min(edge_weights))
    # edge_weights = 0.1 + 3 * (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights))

    nodes_from = np.isin(n_path['bodyId'], n_source['bodyId'])
    nodes_to = np.isin(n_path['bodyId'], n_target['bodyId'])

    # # append a number to distinguish same instances
    # node_label = n_path['instance'] + '_' + n_path.groupby('instance').cumcount().astype(int).astype(str)
    # # remove "_0"
    # node_label = node_label.str.replace('_0', '')
    # append bodyId to distinguish same instances, if no instance, use bodyId
    node_label = n_path['instance'].fillna('') + '_' + n_path['bodyId'].astype(str)
    mapping = dict(zip(np.arange(len(n_path)), node_label))

    node_color = ['magenta' if nodes_from[i] else ('orange' if nodes_to[i] else 'steelblue') for i in range(len(n_path))]
    node_shape = ['triangle' if nodes_from[i] else ('square' if nodes_to[i] else 'dot') for i in range(len(n_path))]
    #node_size = np.sum(m_adj, axis=0) + np.sum(m_adj, axis=1)
    node_name = n_path['instance'].fillna('') + '_' + n_path['bodyId'].astype(str)
    node_id = n_path['bodyId']

    # create networkx directed graph
    id_path = n_path['bodyId'].values
    net = nx.from_numpy_array(m_adj.reindex(index=id_path).reindex(columns=id_path).to_numpy(), create_using=nx.DiGraph)
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), 50*node_size)), 'size')  # dict(net.degree)
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), node_color)), 'color')
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), node_shape)), 'shape')
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), node_name)), 'name')
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), node_name)), 'title')
    nx.set_node_attributes(net, dict(zip(np.arange(len(n_path)), node_id)), 'bodyId')
    net = nx.relabel_nodes(net, mapping)

    print('Graphing the network..')
    net2 = Network(directed=True, layout=False)
    net2.height = 1000 #"900px" "75%"
    net2.width = 1000
    net2.from_nx(net)

    # for i in range(len(node_name)):
    #     net2.nodes[i]['title'] = node_name[i] # to show bodyId on hover over nodes
        # specify [x y], using the largest layer value as x, y as the order in ids_layer
    # list of bodyIds
    node_bodyid = [x.get('bodyId') for x in net2.nodes]
    node_bodyid = np.array(node_bodyid)
    if ids_layer is not None:
        # 0.1 per layer, convert to int
        ids_layer['layer'] = (ids_layer['layer'] * 10).astype(int)
        # number of layers
        N = max(ids_layer['layer'])
        layer_w = net2.width / (N+1)
        # assign x, y
        for i in range(ids_layer['layer'].max()+1):
            ids = ids_layer[ids_layer['layer']==i]['bodyId']
            if not ids.empty:
                layer_h = net2.height / (len(ids)+1)
                for j, k in enumerate(ids):
                    # find which node has this bodyId
                    ind_node = np.where(node_bodyid == k)[0][0]

                    net2.nodes[ind_node]['x'] = (0.5 + i) *layer_w
                    net2.nodes[ind_node]['y'] = (0.5 + j) *layer_h

    net2.toggle_physics(False)
    net2.show_buttons(filter_=['node', 'edge', 'physics'])

    if save_path is not None:
        # chech if it is a windows path
        if isinstance(save_path, Path):
            save_path = str(save_path)
        # net2.save_graph(save_path)
        net2.show(save_path, notebook=False)
    else:
        HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
        net2.show(str(HTML_FIG_DIR / "graph_pyvis.html"), notebook=False)

    return True


# === From main src/quan_propagation/plotting_functions.py (primitives family) ===

# (body from Judith — formatting / signature diff)
def plot_directedness_triangle(plot_df, c_col, t_col, cmap, t_stars=[], height=450, width=450):

    fig = px.scatter_ternary(plot_df, a="frac_la", b="frac_fb", c="frac_ff", hover_name=t_col, \
                            color=c_col, color_discrete_map=cmap)
    fig.update_traces(marker=dict(size=7, opacity=0.4))
    for t in t_stars:
        example_sub = plot_df[plot_df[t_col]==t]
        fig.add_trace(go.Scatterternary(
            a=example_sub['frac_la'].values, 
            b=example_sub['frac_fb'].values,
            c=example_sub['frac_ff'].values,
            mode='markers+text',
            marker=dict(size=7, color=cmap[example_sub[c_col].values[0]], opacity=1, 
                        line=dict(color='black', width=1)),
            text=[s.split('_')[0] for s in example_sub[t_col]],
            textposition='middle left'
        ))
    fig.update_layout(
        plot_bgcolor="white",  
        paper_bgcolor="white", 
        font=dict(family='arial', size=18),
        ternary=dict(
            sum=1,
            aaxis=dict(title="la", showgrid=False, showline=True, linecolor="black", tickvals=[0,0.5,1]),
            baxis=dict(title="fb", showgrid=False, showline=True, linecolor="black", tickvals=[0,0.5,1]),
            caxis=dict(title="ff", showgrid=False, showline=True, linecolor="black", tickvals=[0,0.5,1]),
        ),
        height=height,
        width=width,
        showlegend=False,
    )

    return fig


def plot_gaussian_params(params_df, example_bid=None, height=500, width=500, fac=1):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=params_df['x0'].values,
        y=fac*params_df['y0'].values,
        mode='markers',
        marker=dict(color='red', size=6, symbol='cross'),
        showlegend=False,
    ))    
    for i in range(params_df.shape[0]):
        params_single_df = params_df.iloc[i]
        if params_single_df.bodyId==example_bid:
            line_width=4
            fig.add_trace(go.Scatter(
                x=[params_single_df['x0']],
                y=[fac*params_single_df['y0']],
                mode='markers',
                marker=dict(color='red', size=10, symbol='cross'),
                showlegend=False,
            ))    
        else:
            line_width=1
        fig.add_trace(go.Scatter(
            x=params_single_df['x0'] +\
                params_single_df['a']*np.cos(params_single_df['phi'])*np.cos(np.linspace(0, 2*np.pi, 20)) - \
                params_single_df['b']*np.sin(params_single_df['phi'])*np.sin(np.linspace(0, 2*np.pi, 20)), 
            y=fac*params_single_df['y0'] +\
                fac*params_single_df['b']*np.cos(params_single_df['phi'])*np.sin(np.linspace(0, 2*np.pi, 20)) + \
                fac*params_single_df['a']*np.sin(params_single_df['phi'])*np.cos(np.linspace(0, 2*np.pi, 20)), 
            mode='lines', 
            line=dict(color='red', width=line_width),
            showlegend=False,
        ))
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.update_layout(
        font={
            "family":'Arial'
            , "size" : 14
        }
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=500
        , width=width
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_stacked_bars(plot_df, x_col, y_col, c_col, cmap, height=350, width=500, xrot=0):

    fig = px.bar(plot_df,
                x=x_col,
                y=y_col,
                color=c_col,
                color_discrete_map=cmap,
                category_orders={c_col: list(cmap.keys())}
    )
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.add_hline(y=1, line_color='black', line_width=1)
    fig.update_layout(
        boxmode='group', 
        xaxis_tickangle=xrot,
        yaxis_range=[-0.01,1.01],
        font=dict(family='arial', size=18),
        height=height,
        width=width,
        plot_bgcolor = 'white',
        paper_bgcolor = 'white',
    )
    fig.update_xaxes(showgrid=False, tickvals=plot_df[x_col].values)
    fig.update_yaxes(showgrid=False, tickvals=[0,1])

    return fig


# (body from Judith — formatting / signature diff)
def plot_heatmatrix(mat, tick_labels, cmap='Greys', height=500, width=500, anno=None):

    n_plot = len(tick_labels)

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=tick_labels,
        y=tick_labels,
        colorscale=cmap,
        reversescale=False,  # Set to False to invert colormap (dark = high values)
        showscale=True,
        text=anno,
        texttemplate="%{text}",
        textfont={"size":18}),
    )
    fig.add_shape(
        type="line",
        x0=-0.5, y0=-0.5, x1=n_plot-0.5, y1=n_plot-0.5,
        line=dict(color="gray", width=2)
    )
    fig.update_layout(
        xaxis_title='post layer',
        yaxis_title='pre layer',
        xaxis_tickvals=list(range(n_plot)),
        xaxis_ticktext=tick_labels,
        yaxis_tickvals=list(range(n_plot)),
        yaxis_ticktext=tick_labels,
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
        font=dict(family='arial', size=18),
        height=height,
        width=width, 
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.update_xaxes(
        showline=True,
        title_standoff=0,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )
    fig.update_yaxes(
        showline=True,
        title_standoff=2,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )

    return fig


def plot_patterns(plot_df, y_col, cols, patterns, height=600, width=400):

    j_max = plot_df[cols].idxmax(1)

    fig = go.Figure()
    for j in range(len(cols)):
        # For stacked bars, we add one trace per part
        fig.add_trace(go.Bar(
            y=plot_df[y_col],
            x=plot_df[cols[j]],
            orientation='h',
            name=f"{cols[j]}",
            marker=dict(
                color="white",
                line=dict(color="black", width=1+(j_max==cols[j])),
                pattern=dict(shape=patterns[j])
            )
        ))
    fig.add_trace(go.Bar(
        y=plot_df[y_col],
        x=[0]*len(plot_df),  # zero-width bars
        orientation='h',
        marker=dict(color="white", line=dict(color="black", width=1)),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Bar(
        y=plot_df[y_col],
        x=[1e-9]*len(plot_df),
        base=1 - 1e-9,
        orientation="h",
        marker=dict(color="white", line=dict(color="black", width=1)),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(
        barmode="stack",
        bargap=0.3,
        height=height,  
        width=width,
        xaxis=dict(title="fraction", range=[-0.01, 1.01]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_box_scatter(plot_df, x_col, y_col, c_col, color_df, height=350, width=450):

    all_groups = plot_df[c_col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = plot_df[plot_df[c_col]==group]
        fig.add_trace(go.Box(x=sub_df[x_col], y=sub_df[y_col], name=str(group), \
                            marker=dict(color=color_df.loc[group,'color']), \
                            line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        xaxis_title=x_col
        , yaxis_title=y_col
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , font=dict(family='arial', size=18)
        , height=height
        , width=width
        , boxmode='group'
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_box(plot_df, x_col, y_col, c_col, cmap, height=600, width=400):

    fig = px.box(plot_df, x=x_col, y=y_col, points="all", 
             color=c_col, color_discrete_map=cmap)
    fig.update_layout(
         font=dict(family='arial', size=18)
       , xaxis_title=x_col
        , yaxis_title=y_col
        , xaxis_range=[0.9*plot_df[x_col].min(), 1.1*plot_df[x_col].max()]
        , yaxis_tickvals=plot_df[y_col].values
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
        , showlegend=False
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_point_line(plot_df, x_col, y_col, x_thre=None, y_thre=None, height=500, width=600, color='black'):

    fig = go.Figure()
    if x_thre is not None:
        fig.add_vline(x=x_thre, line_width=1, line_dash="dash", line_color="gray")
    if y_thre is not None:
        fig.add_hline(y=y_thre, line_width=1, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(
        x=plot_df[x_col].values, 
        y=plot_df[y_col].values, 
        mode='markers+lines',
        marker=dict(size=8, color=color, opacity=0.7),
        line=dict(width=1, color=color),
    ))
    fig.update_layout(
        font=dict(family='arial', size=18),
        xaxis_title=x_col,
        yaxis_title=y_col,
        xaxis_range=[0.9*plot_df[x_col].min(), 1.1*plot_df[x_col].max()],
        yaxis_range=[0.9*plot_df[y_col].min(), 1.1*plot_df[y_col].max()],
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
        height=height,
        width=width,
        showlegend=False
    )
    fig.update_xaxes(
        showline=True,
        title_standoff=0,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )
    fig.update_yaxes(
        showline=True,
        title_standoff=2,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside'
        , ticklen=5
        , tickcolor='black'        
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_scatter(plot_df, x_col, y_col, c_col, t_col, color_df, t_stars=[], 
                 height=400, width=400, marker='circle', size=7, show_labels=True):

    fig = go.Figure(go.Scatter(
        x=plot_df[x_col].values, 
        y=plot_df[y_col].values,
        mode='markers',
        marker=dict(size=size, color=color_df.loc[plot_df[c_col].values,'color'].values, opacity=0.4, 
                    symbol=marker),
        text=plot_df[t_col]
    ))
    for t in t_stars:
        example_sub = plot_df[plot_df[t_col]==t]
        fig.add_trace(go.Scatter(
            x=example_sub[x_col].values, 
            y=example_sub[y_col].values,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=size, color=color_df.loc[example_sub[c_col].values,'color'].values, opacity=1, 
                        symbol=marker, line=dict(color='black', width=1)),
            # text=[s.split('_')[0] for s in example_sub[t_col]],
            text=[s.replace('_R','') for s in example_sub[t_col]] if show_labels else None,
            textposition='bottom center'
        ))
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=width, height=height,
        font=dict(family='arial', size=18),
        showlegend=False,
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',\
        )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'
    )

    return fig


# (body from Judith — formatting / signature diff)
def plot_hist_sum_lines(df, val, sum_val, col, hist_bins, color_df, height=400, width=500, norm=True):
    """
    Plots histogram lines where the y-value is the sum of another column within each bin.
  
    """

    plot_df = df.copy()
    all_groups = plot_df[col].unique()
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plot_df['val_bin'] = pd.cut(plot_df[val], bins=hist_bins, labels=bin_centers).astype(float)

    fig = go.Figure()
    for group in all_groups:
        sub_df = plot_df[plot_df[col]==group]
        hist_bin_df = sub_df.groupby('val_bin', observed=False)[sum_val].sum()
        hist_bin_df = hist_bin_df.reindex(bin_centers, fill_value=0).reset_index()
        hist_bin_df.columns = ['bin', 'count']
        if norm:
            hist_bin_df['count'] = hist_bin_df['count']/hist_bin_df['count'].sum()
        fig.add_trace(go.Scatter(x=hist_bin_df['bin'], y=hist_bin_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color']),
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=18)
        , xaxis_title=val
        , xaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    
    return fig


# (body from Judith — formatting / signature diff)
def plot_hist_lines_v(df, val, col, hist_bins, color_df, height=400, width=500, norm=True):

    all_groups = df[col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = df[df[col]==group]
        hist_counts, _ = np.histogram(sub_df[val], bins=hist_bins)
        if norm:
            hist_counts = hist_counts / hist_counts.sum()
        hist_df = pd.DataFrame({'bin': (hist_bins[:-1]+hist_bins[1:])/2, 'count': hist_counts})
        fig.add_trace(go.Scatter(y=hist_df['bin'], x=hist_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color']),
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=18)
        , yaxis_title=val
        , yaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    
    return fig


# (body from Judith — formatting / signature diff)
def plot_hist_lines(df, val, col, hist_bins, color_df, height=400, width=500, norm=True):

    all_groups = df[col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = df[df[col]==group]
        hist_counts, _ = np.histogram(sub_df[val], bins=hist_bins)
        if norm:
            hist_counts = hist_counts / hist_counts.sum()
        hist_df = pd.DataFrame({'bin': (hist_bins[:-1]+hist_bins[1:])/2, 'count': hist_counts})
        fig.add_trace(go.Scatter(x=hist_df['bin'], y=hist_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color']),
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=18)
        , xaxis_title=val
        , xaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    
    return fig


# (body from Judith — formatting / signature diff)
def plot_pie_chart(df, val, col, cmap, height=400, width=400, title=''):

    fig = px.pie(df, values=val, names=col, title=title,
                color=col, color_discrete_map=cmap)
    fig.update_traces(textposition='inside', texttemplate='%{percent:.0%}')
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor = 'white',
        paper_bgcolor = 'white',
        font=dict(family='arial', size=18)
    )

    return fig


# === Judith-unique primitives (from src/plotting_functions.py) ===

def plot_eyesymbol(color_segments):

    # Parameters
    a, b = 1.2, 2   # major/minor axes for large ellipse
    a2, b2 = 0.5, 0.8  # smaller inner ellipse
    N = 200
    theta = np.linspace(-np.pi/4, 7*np.pi/4, N)

    # Split into 4 sectors (each 90°)
    sectors = [(-np.pi/4, np.pi/4), (np.pi/4, 3*np.pi/4), (3*np.pi/4, 5*np.pi/4), (5*np.pi/4, 7*np.pi/4)]
    colors = [color_segments[2],color_segments[1],color_segments[0],color_segments[3]]
    fig = go.Figure()

    # Add 4 colored outer segments
    for (start, end), color in zip(sectors, colors):
        mask = (theta >= start) & (theta <= end + 2*np.pi/N)
        x = np.concatenate(([0], a*np.cos(theta[mask]), [0]))
        y = np.concatenate(([0], b*np.sin(theta[mask]), [0]))
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill='toself',
            fillcolor=color,
            line=dict(color='black', width=1),
            mode='lines',
            showlegend=False
        ))

    # Add small central ellipse
    x_inner = a2 * np.cos(theta)
    y_inner = b2 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_inner, y=y_inner,
        fill='toself',
        fillcolor=color_segments[4],
        line=dict(color='black', width=1.5),
        mode='lines',
        showlegend=False
    ))

    # Style layout
    fig.update_layout(
        width=500, height=500,
        xaxis=dict(scaleanchor='y', visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def plot_hist_cumsum_lines(df, val, col, hist_bins, color_df, height=500, width=400):

    all_groups = df[col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = df[df[col]==group]
        hist_counts, _ = np.histogram(sub_df[val], bins=hist_bins)
        hist_counts = hist_counts / hist_counts.sum()
        hist_df = pd.DataFrame({'bin': (hist_bins[:-1]+hist_bins[1:])/2, 'count': np.cumsum(hist_counts)})
        fig.add_trace(go.Scatter(x=hist_df['bin'], y=hist_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color']),
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=18)
        , xaxis_title=val
        , yaxis_title='cum. frac.'
        , xaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    
    return fig


def plot_hist_cumsum_lines_v(df, val, col, hist_bins, color_df, height=500, width=400):

    all_groups = df[col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = df[df[col]==group]
        hist_counts, _ = np.histogram(sub_df[val], bins=hist_bins)
        hist_counts = hist_counts / hist_counts.sum()
        hist_df = pd.DataFrame({'bin': (hist_bins[:-1]+hist_bins[1:])/2, 'count': np.cumsum(hist_counts)})
        fig.add_trace(go.Scatter(y=hist_df['bin'], x=hist_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color']),
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=18)
        , yaxis_title=val
        , xaxis_title='cum. frac.'
        , yaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickcolor='black'  
    )
    
    return fig
