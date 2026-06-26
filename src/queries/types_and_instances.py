import pandas as pd

from neuprint import Client, fetch_custom, fetch_neurons, NeuronCriteria as NC
from neuprint.client import inject_client
import numpy as np


def fetch_input_hex_ids(
      input_instance:str
    , output_instance:str
    , neuropil:str
    , batch_size=100
) -> pd.DataFrame:
    """
    Find hex ids of pre synapses between two cell types

    Parameters
    ----------
    input_instance : str
        instance of input
    output_instance : str
        instance of output
    neuropil : str
        neuropil roi
    client : neuprint.Client
        Client used for the connection. If no explicit client is provided, then the `defaultclient`
        is used.

    Returns
    -------
    df : pd.DataFrame
        bodyId_pre : int
            input cell
        bodyId_post : int
            output cell
        hex1_id : int
            first coordinate of hex coordinate
        hex2_id : int
            second coordinate of hex coordinate
    """

    neurons_df, _ = fetch_neurons(NC(instance=output_instance))
    bodyId_arr = neurons_df['bodyId'].unique()
    n_batches = int(np.ceil(neurons_df['bodyId'].nunique()/batch_size))
    
    hex_df = pd.DataFrame()
    for i in range(n_batches):
        bids = list(bodyId_arr[i*batch_size:(i+1)*batch_size])
        
        cql = f"""
            MATCH (a:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(as:Synapse)
            -[:`SynapsesTo`]->(:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(b:Neuron)
            WHERE a.instance = '{input_instance}' 
            AND b.bodyId IN {bids}
            AND as['{neuropil}'] IS NOT NULL
            AND (exists(as.olHex1) AND as.olHex1 IS NOT NULL)
            AND (exists(as.olHex2) AND as.olHex2 IS NOT NULL)
            WITH DISTINCT a, as, b
            RETURN 
                a.bodyId as bodyId_pre
              , b.bodyId as bodyId_post
              , as.olHex1 as hex1_id
              , as.olHex2 as hex2_id
        """
        df = fetch_custom(cql)
        
        if i==0:
            hex_df = df
        else:
            hex_df = pd.concat([hex_df, df], sort=False)        

    return hex_df


@inject_client
def fetch_ol_types_and_instances(
    *
  , include_placeholders:bool=False
  , side:str="R-dominant"
  , return_type:str="dataframe"
  , client:Client=None
) -> pd.DataFrame | list[str]:
    """
    Code copied from optic-lobe-connectome/src/queries/completeness.py
    
    Get a list of named types and instances from the optic lobe.

    Parameters
    ----------
    side : str, default = 'R-dominant'
        options include, 'R', 'L', 'R-dominant' or 'both'.
        'R' means all neurons that have their cellbody on the right side, 'L' means that their
        cellbody is on the left side, 'R-dominant' chooses the neurons that have their 'dominant features'
        in the right hemisphere, and 'both' means to get both sides (if available).
        For most analysis that works on one side, the 'R-dominant' is probably the best choice. There will
        be a 'L-dominant' once the other side is proof-read. 'both' returns the types that are postsent on
        either side and counts their total. If you know what you are doing and there is a reason to diverge,
        you can choose 'R' or 'L'.
    include_placeholders : bool, default=False
        Include cell types that are intended as placeholders in the results (e.g. ME_VPN). Not
        recommended, therefore defaults to false.
    return_type : str, default='dataframe'
        defines what data type the function returns.
    client : neuprint.Client
            Client used for the connection. If no explicit client is provided, then the
            `defaultclient` is used.

    Returns
    -------
    named_df : pandas.DataFrame
        type : str
            type name
        instance : str
            instance name
        count : int
            number of neurons in that instance
    """

    assert return_type in ["dataframe", "list"],\
        f"Wrong return type '{type}', only 'dataframe' and 'list' are allowed"

    assert side in ["R", "L", "R-dominant", "both"],\
        f"Unsupported side '{side}', only 'R', 'L', 'R-dominant' or 'both' are allowed"

    str_ignore = ""
    str_side = ""
    str_dominant = ""

    if not include_placeholders:
        str_ignore = f"""
            AND NOT n.type ENDS WITH '_unclear'
            AND NOT n.instance CONTAINS 'unclear'
            AND n.type<>'Pm7_Li28'
        """

    if side in ['R', 'L']:
        str_side = f"AND n.instance ENDS WITH '_{side}'"

    if side in ['R-dominant', 'L-dominant']:
        side_char = side[0]
        str_dominant = f"""
            WITH DISTINCT n.type AS type
              , count(distinct n.instance) AS instance_count
              , collect(n) AS ns
            UNWIND ns AS n
            WITH n, instance_count
            WHERE instance_count <=1 OR n.instance ENDS WITH '_{side_char}'
        """

    cql = f"""
        MATCH(n:Neuron)
        WHERE (
                n.`LA(R)`=True 
                OR n.`ME(R)`=True 
                OR n.`LO(R)`=True
                OR n.`LOP(R)`=True
                OR n.`AME(R)`=True
            )
            AND (n.type IS NOT NULL and n.type <> '')
            AND (n.instance IS NOT NULL and n.instance <> ''
            {str_ignore}
            {str_side})
        {str_dominant}
        RETURN distinct n.type as type, n.instance as instance, count(n.bodyId) as count
        ORDER BY toLower(type), count DESC
    """

    named_df = client.fetch_custom(cql)

    if return_type == "list":
        return named_df.loc[:, "type"].to_list()
    return named_df
    

def fetch_input_neuropil_df(
    df:pd.DataFrame
  , threshold:float=0.05
):
    """
    Code modified from the function get_neuropil_df
    in optic-lobe-connectome/src/utils/make_overall_summary_fl.py

    Find the input neuropil of specified cell instances.

    Parameters
    ----------
    df : pd.DataFrame
        data with all the cell instances
    threshold : float
        threshold fraction of synapses that needs to be crossed within a neuropil
        to be assigned to that brain region

    Returns
    -------
    neuropil_df : pd.DataFrame
        roi : str
            neuropil for which synapse fraction is above threshold
        instance : str
            cell instance
        syn_frac : float
            synapse fraction in that neuropil
    """

    assert 0<= threshold <= 1, f"Threshold must be between 0 and 1, not {threshold}"

    cql_cell_types=f"""
        UNWIND {df['instance'].to_list()} as instance
        UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
        MATCH (n:Neuron)
        WHERE n.instance = instance
        WITH 
            distinct n
          , apoc.convert.fromJsonMap(n.roiInfo) as nri
          , coalesce(n.post, 0) as syn_total
          , roi
          , instance
        with
            coalesce(nri[roi].post, 0) as syn_post
          , syn_total
          , roi
          , instance
          , n
        WITH
            distinct n.type as type
          , roi
          , instance
          , sum(syn_post) as syn_post
          , sum(syn_total) as syn_total

        WITH 
          CASE 
                WHEN syn_total> 0 THEN toFloat(syn_post)/ syn_total
                ELSE 0 
            END AS syn_frac
          , roi
          , instance
        WHERE syn_frac >= {threshold}
        RETURN 
            distinct roi
          , instance
          , syn_frac
    """

    celltype_df = fetch_custom(cql_cell_types)

    return celltype_df
    

def fetch_top_input_instances(
    instance_name:str
  , neuropil:str
  , limit:int=None
) -> pd.DataFrame:
    """
    Code modified from the function __get_input_output 
    in optic-lobe-connectome/src/utils/instance_summary.py
    """
    
    assert limit is None or (0 < limit and isinstance(limit, int)),\
        f"If limit is defined, it must be an integer > 0, not {limit} of type {type(limit)}"

    cql = f"""
        MATCH (n:Neuron)<-[e:ConnectsTo]-(m:Neuron)
        WHERE n.instance = '{instance_name}'
        AND NOT m.instance IS NULL AND NOT m.type IS NULL
        AND apoc.convert.fromJsonMap(e.roiInfo)['{neuropil}'].post > 0
        WITH apoc.convert.fromJsonMap(e.roiInfo) as nri
        WITH sum(coalesce(nri['{neuropil}'].post, 0)) as all_syn
        MATCH (n:Neuron)<-[e:ConnectsTo]-(m:Neuron)
        WHERE n.instance = '{instance_name}'
        AND NOT m.instance IS NULL AND NOT m.type IS NULL
        AND apoc.convert.fromJsonMap(e.roiInfo)['{neuropil}'].post > 0
        WITH apoc.convert.fromJsonMap(e.roiInfo) as nri, m.instance as instance, all_syn
        WITH sum(coalesce(nri['{neuropil}'].post, 0)) as wgt, instance, all_syn
        RETURN instance, toFloat(wgt)/all_syn as frac
        order by wgt DESC
        LIMIT {limit}
    """ 
    ret = fetch_custom(cql)
    ret['cum_frac'] = ret['frac'].cumsum()
    
    return ret