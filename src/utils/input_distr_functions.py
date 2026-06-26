from pathlib import Path
from dotenv import find_dotenv
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import entropy
import csv

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses, fetch_synapse_connections
from utils.hex_geometry import find_hex_ids, find_neuron_hex_ids
from queries.types_and_instances import fetch_top_input_instances, fetch_input_neuropil_df, fetch_input_hex_ids
from utils.plotting_functions import generate_random_color, plot_1d_polar, plot_1d_distr
from utils.ol_types import OLTypes


def fetch_n_find_rel_hex(
    hex_red_df
  , input_instance
  , output_instance
  , neuropil
  , batch_size=10
) -> pd.DataFrame:

    #assign hex_ids to synapses
    try:
        syn_input_df = fetch_input_hex_ids(input_instance, output_instance, neuropil, batch_size=batch_size)
    except Exception as e:
        print(f'Exception for {input_instance} inputting into {output_instance}')
        return pd.DataFrame
    
    #attach home column index and compute difference
    hex_diff_df = syn_input_df.merge(hex_red_df, on='bodyId_post', suffixes=('_pre','_post'))
    hex_diff_df['hex1_id'] = hex_diff_df['hex1_id_pre']-hex_diff_df['hex1_id_post']
    hex_diff_df['hex2_id'] = hex_diff_df['hex2_id_pre']-hex_diff_df['hex2_id_post']
    
    return hex_diff_df


def find_rel_hex(
    hex_red_df
  , syn_input_df
  , neuropil
) -> pd.DataFrame:

    #assign hex_ids to synapses
    hex_input_df = find_hex_ids(syn_input_df, roi_str=neuropil)
    syn_input_df = syn_input_df.join(hex_input_df)
    
    #attach home column index and compute difference
    hex_diff_df = syn_input_df.merge(hex_red_df, on='bodyId_post', suffixes=('_pre','_post'))
    hex_diff_df['hex1_id'] = hex_diff_df['hex1_id_pre']-hex_diff_df['hex1_id_post']
    hex_diff_df['hex2_id'] = hex_diff_df['hex2_id_pre']-hex_diff_df['hex2_id_post']
    
    return hex_diff_df


def find_1d_input_distrs(input_df):

    data_df = input_df.copy()
    data_df['radius'] = np.sqrt(data_df['hex1_id'].values**2+data_df['hex2_id'].values**2)
    
    rad_max = data_df['radius'].max()
    rad_bins = np.arange(-0.25, int(np.round(rad_max))+1.25, 0.5)
    ind_rad = np.digitize(data_df['radius'].values, bins=rad_bins)
    rad_count = np.zeros(rad_bins.shape[0]-1)
    for i in range(rad_bins.shape[0]-1):
        rad_count[i] = data_df.iloc[ind_rad==i+1]['frac'].sum()
    rad_df = pd.DataFrame({'radius': (rad_bins[:-1]+rad_bins[1:])/2, \
                           'frac': rad_count/rad_count.sum()})
    
    #angles between [0,360]
    data_df['angle'] = np.arctan2(data_df['hex2_id'].values, data_df['hex1_id'].values)
    idx_neg = data_df[data_df['angle']<0].index
    data_df.loc[idx_neg, 'angle'] = data_df.loc[idx_neg, 'angle'] + 2*np.pi
    data_df['angle'] = data_df['angle']*180/np.pi
    rad_nonzero_df = data_df[ (data_df['radius']>0.5) ] 

    ang_bins = np.arange(0,361,45)-45/2
    ind_ang = np.digitize(rad_nonzero_df['angle'].values, bins=ang_bins)
    ang_count = np.zeros(ang_bins.shape[0]-1)
    for i in range(ang_bins.shape[0]-1):
        ang_count[i] = rad_nonzero_df.iloc[ind_ang==i+1]['frac'].sum()
    ang_df = pd.DataFrame({'angle': np.hstack([(ang_bins[:-1]+ang_bins[1:])/2, 0]), \
                          'frac': np.hstack([ang_count/ang_count.sum(), ang_count[0]/ang_count.sum()])})

    return rad_df, ang_df


def store_input_of_ol(limit, thre, instance_list=[]):

    ol = OLTypes()
    types = ol.get_neuron_list()
    cell_instances_df = types[["type","hemisphere"]].agg("_".join, axis=1).to_frame(name='instance')
    if len(instance_list)>0:
        cell_instances_df = cell_instances_df[np.isin(cell_instances_df['instance'],instance_list)]

    neuropil_df = fetch_input_neuropil_df(cell_instances_df,threshold=thre)
    neuropil_df = neuropil_df[np.isin(neuropil_df['roi'], ['ME(R)','LO(R)','LOP(R)'])].reset_index(drop=True)
    neuropil_df['syn_frac'] = ['%.4f'%s for s in neuropil_df['syn_frac'].values]

    data_path = Path(Path(find_dotenv()).parent, 'results', 'input_distribution')
    data_path.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    if len(instance_list)>0:
        data_fn = data_path / f'multi_top{limit}_top{limit}_thre{thre}_{date}.csv'
    else:
        data_fn = data_path / f'all_top{limit}_top{limit}_thre{thre}_{date}.csv'
    #delete file data_fn if it already exists
    if os.path.isfile(data_fn):
        os.remove(data_fn)
    with open(data_fn,'w+',newline='') as my_csv:
        csv_writer=csv.writer(my_csv)
        header = np.array([['target instance'],['roi'],['frac in roi'],['input instance 1'],['kl rad 1'],['kl ang 1']\
                              ,['input instance 2'],['kl rad 2'],['kl ang 2'],['input instance 3'],['kl rad 3'],['kl ang 3']\
                                ,['input instance 4'],['kl rad 4'],['kl ang 4'],['input instance 5'],['kl rad 5'],['kl ang 5']]).T
        csv_writer.writerows(header)

    if limit==5:
        colors_all = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    else:
        colors_all = None

    for i in range(neuropil_df.shape[0]):
        data_row = store_1d_input_distrs(neuropil_df.loc[i,'instance'], neuropil_df.loc[i,'roi'], limit, colors_all)
        data_row.insert(0, neuropil_df.loc[i,'syn_frac'])
        data_row.insert(0, neuropil_df.loc[i,'roi'])
        data_row.insert(0, neuropil_df.loc[i,'instance'])
        
        #write csv's, one row at a time            
        with open(data_fn,'a+',newline='') as my_csv:
            csv_writer=csv.writer(my_csv)
            csv_writer.writerows([data_row])


def store_1d_input_distrs(instance_name, neuropil, limit, colors_all=None):

    #colors for plotting
    if colors_all==None:
        colors_all = [generate_random_color() for _ in range(limit)]
        colors_all.insert(0, '#808080')
    #upper threshold on number of hex coordinates in 1d
    hex0 = 45
    
    #get home columns
    syn_df = fetch_synapses(NC(instance=instance_name), SC(rois=[neuropil], type='post'))
    hex_df = find_neuron_hex_ids(syn_df, roi_str=neuropil, method='COM')
    hex_red_df = hex_df[['bodyId','hex1_id','hex2_id']]\
        .rename(columns={'bodyId': 'bodyId_post'})
    
    #control: use all post synapses
    conn_control_df = fetch_synapse_connections(None, \
                                        NC(instance=instance_name) ,\
                                        SC(rois=[neuropil]))
    syn_control_df = conn_control_df[['bodyId_pre','x_post','y_post','z_post','bodyId_post']]\
        .rename(columns={'x_post': 'x', 'y_post': 'y', 'z_post': 'z'})
    hex_control_df = find_rel_hex(hex_red_df, syn_control_df, neuropil)

    #count fraction of synapses in columns
    hex_control_df = hex_control_df.groupby(['bodyId_post','hex1_id','hex2_id'])['bodyId_pre'].count().to_frame(name='count').reset_index()
    hex_control_df['frac'] = hex_control_df.groupby('bodyId_post')['count'].apply(lambda x: x/x.sum()).values
    hex_control_df['hex_names'] = 100*(hex_control_df['hex1_id']+hex0) + hex_control_df['hex2_id']+hex0
    mean_control_df = hex_control_df.groupby(['bodyId_post','hex_names'])['frac'].first().unstack(-1, 0).mean(0).to_frame(name='frac').reset_index()
    mean_control_df['hex1_id'] = (mean_control_df['hex_names'].values/100).astype(int)-hex0
    mean_control_df['hex2_id'] = mean_control_df['hex_names']-100*(mean_control_df['hex1_id']+hex0)-hex0    
    
    rad_dist_control_df, angle_dist_control_df = find_1d_input_distrs(mean_control_df) #hex_control_df)
    
    #make a figure with all 1D radial distributions
    rad_dist_control_df['frac'] = rad_dist_control_df['frac']/rad_dist_control_df['frac'].sum()
    fig_rad = plot_1d_distr(rad_dist_control_df, 'radius', 'frac')
    fig_rad.data[0].marker.color = colors_all[0]
    fig_rad.data[0].line.color = colors_all[0]
    fig_rad.data[0].name = 'control'  
    y_max = max(fig_rad.data[0].y)

    #make a figure with all 1D angular distributions
    if angle_dist_control_df.shape[0]>0:
        plot_data_df = angle_dist_control_df
        plot_data_df['frac'] = plot_data_df['frac']/plot_data_df['frac'].sum()
    else:
        plot_data_df = pd.DataFrame({'angle': [0], 'frac': [0]})
    fig_angle = plot_1d_polar(plot_data_df, 'angle', 'frac')
    fig_angle.data[0].marker.color = colors_all[0]
    fig_angle.data[0].line.color = colors_all[0]
    fig_angle.data[0].name = 'control'
    y_max2 = max(fig_angle.data[0].r)

    #identify strongest input cell types
    input_instances = fetch_top_input_instances(instance_name, neuropil, limit)

    data_row = []
    fig_ctr = 1  
    for input_instance in input_instances['instance'].values:
    
        #fraction of synapses per column
        hex_instance_df = fetch_n_find_rel_hex(hex_red_df, input_instance, instance_name, neuropil)
        if hex_instance_df.empty:
            continue
        data_row.append(input_instance)
            
        hex_instance_df = hex_instance_df.groupby(['bodyId_post','hex1_id','hex2_id'])['bodyId_pre'].count().to_frame(name='count').reset_index()
        hex_instance_df['frac'] = hex_instance_df.groupby('bodyId_post')['count'].apply(lambda x: x/x.sum()).values
        hex_instance_df['hex_names'] = 100*(hex_instance_df['hex1_id']+hex0) + hex_instance_df['hex2_id']+hex0
        mean_df = hex_instance_df.groupby(['bodyId_post','hex_names'])['frac'].first().unstack(-1, 0).mean(0).to_frame(name='frac').reset_index()
        mean_df['hex1_id'] = (mean_df['hex_names'].values/100).astype(int)-hex0
        mean_df['hex2_id'] = mean_df['hex_names']-100*(mean_df['hex1_id']+hex0)-hex0    
    
        rad_dist_df, angle_dist_df = find_1d_input_distrs(mean_df) #hex_instance_df)
        
        #significance: KL divergence to control (normalization into prob distr's is taken care of in entropy function)
        rad_dist_df = rad_dist_df.merge(rad_dist_control_df, on='radius', suffixes=('','_control'))
        rad_pos_df = rad_dist_df[rad_dist_df['frac_control']>0]
        kl_div = entropy(rad_pos_df['frac'].values, rad_pos_df['frac_control'].values)
        data_row.append('%.4f'%kl_div)
        if angle_dist_control_df.shape[0]*angle_dist_df.shape[0]>0:
            angle_dist_df = angle_dist_df.merge(angle_dist_control_df, on='angle', suffixes=('','_control'))
            angle_pos_df = angle_dist_df[angle_dist_df['frac_control']>0]
            kl_div = entropy(angle_pos_df['frac'].values, angle_pos_df['frac_control'].values)
            data_row.append('%.4f'%kl_div)
    
        #add to radial distribution
        rad_dist_df['frac'] = rad_dist_df['frac']/rad_dist_df['frac'].sum()
        fig = plot_1d_distr(rad_dist_df, 'radius', 'frac')
        fig.data[0].marker.color = colors_all[fig_ctr]
        fig.data[0].line.color = colors_all[fig_ctr]
        fig.data[0].name = input_instance
        y_max = max([max(fig.data[0].y), y_max])
        fig_rad.add_traces(fig.data)
        fig_rad.update_layout(annotations=fig_rad.layout.annotations + fig.layout.annotations)
        
        #add to angular distribution
        if angle_dist_df.shape[0]>0:
            angle_dist_df['frac'] = angle_dist_df['frac']/angle_dist_df['frac'].sum()
            fig2 = plot_1d_polar(angle_dist_df, 'angle', 'frac')
            fig2.data[0].marker.color = colors_all[fig_ctr]
            fig2.data[0].line.color = colors_all[fig_ctr]
            fig2.data[0].name = input_instance
            y_max2 = max([max(fig2.data[0].r), y_max2])
            fig_angle.add_traces(fig2.data)
            fig_angle.update_layout(annotations=fig_angle.layout.annotations + fig2.layout.annotations)
        
        fig_ctr += 1

    #save plots
    save_fig_path = Path(Path(find_dotenv()).parent, 'cache', 'input_distribution', 'special cell types')
    save_fig_path.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    
    fig_rad.update_layout(yaxis_range=[0, 1.2*y_max])
    fig_rad.update_layout(title=dict(text=f'Mean radial pre-synapse distribution into {instance_name[:-2]}', x=0.5, y=0.9))    
    fig_rad.write_image(os.path.join(save_fig_path, f'{instance_name}_{neuropil[:-3]}_top{limit}_{date}_radial.pdf'))
    
    fig_angle.update_layout(polar=dict(radialaxis=dict(range=[0, 1.2*y_max2])))
    fig_angle.update_layout(title=dict(text=f'Mean angular pre-synapse distribution into {instance_name[:-2]}', x=0.5, y=0.9))    
    fig_angle.write_image(os.path.join(save_fig_path, f'{instance_name}_{neuropil[:-3]}_top{limit}_{date}_angular.pdf'))

    return data_row

