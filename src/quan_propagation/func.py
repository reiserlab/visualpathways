
import pandas as pd
import numpy as np
import re
import alphashape
import shapely
import warnings
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.spatial import Delaunay, ConvexHull
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import plotly.graph_objects as go

import neuprint
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_adjacencies
import navis
import navis.interfaces.neuprint as neu



def get_roi_outline(roi_names, alpha=0.002, lop=None, zranges=None):
    """
        Generate 2D outlines for a list of ROIs using alpha shapes.

        Parameters:
        roi_names: list of str
            List of ROI names to process.
        alpha: float
            Alpha value for the alpha shape algorithm.
        lop: str, optional
            Line of projection, assuming z-axis, rotate [x y z] if it's not z-axis.
        zranges: list of float, optional
            List of z-ranges to filter the ROIs. Assuming after rotation to have aligned with lop.

        Returns:
        list: 
            List of outlines for the given ROIs.
        """
    if not isinstance(roi_names, list) or not all(isinstance(roi, str) for roi in roi_names):
        raise ValueError("roi_names must be a list of strings.")
    
    # check lop 
    if lop is not None:
        if lop not in ["x", "y", "z"]:
            raise ValueError("lop must be 'x', 'y', or 'z'.")

    # outlines_roi = []
    outlines_roi = [[] for _ in range(len(roi_names))]
    for i, roi in enumerate(roi_names):
        try:
            roi_data = neu.fetch_roi(roi)
        except Exception as e:
            raise RuntimeError(f"Error fetching ROI '{roi}': {e}")
        # [x y z] coordinates roi mesh points
        xyz = roi_data.vertices

        # rotate such that the line of projection is z-axis
        if lop == "x":            # rotate [x y z] to [y z x]
            xyz = xyz[:, [1, 2, 0]]
        elif lop == "y":            # rotate [x y z] to [x z y]
            xyz = xyz[:, [0, 2, 1]]

        if zranges is not None:
            if not isinstance(zranges, list) or not all(isinstance(section, (int, float)) for section in zranges):
                raise ValueError("zranges must be a list of numeric values (integers or floats).")
            if zranges != sorted(zranges) and zranges != sorted(zranges, reverse=True):
                raise ValueError("zranges must be a monotonic (sorted) list of integers (either increasing or decreasing).")
            # section the roi mesh points
            outlines_tmp = [[] for _ in range(len(zranges)-1)]
            for j in range(len(zranges)-1):
                xy = xyz[(xyz[:, 2] > zranges[j]) & (xyz[:, 2] < zranges[j+1]), :2]  # Select the first two columns (x and y)
                # check if xy is not empty
                if xy.shape[0] != 0:
                    alpha_shape = alphashape.alphashape(xy, alpha=alpha)
                    # Handle possible geometry types
                    if alpha_shape is None:
                        outline_coords = []
                    elif hasattr(alpha_shape, 'geoms'):  # MultiPolygon or collection
                        warnings.warn(f"Multi-part geometry for ROI '{roi}' in section {j}; using largest polygon.")
                        outline_coords = []
                        # pick largest polygon exterior to represent outline
                        largest = max(alpha_shape.geoms, key=lambda g: g.area)
                        if not largest.is_empty:
                            outline_coords = [[xv, yv] for xv, yv in largest.exterior.coords]
                    else:  # Single Polygon
                        if alpha_shape.is_empty:
                            outline_coords = []
                        else:
                            outline_coords = [[xv, yv] for xv, yv in alpha_shape.exterior.coords]
                    outlines_tmp[j] = outline_coords
            outlines_roi[i] = outlines_tmp
        else:
            alpha_shape = alphashape.alphashape(points=xyz[:, :2], alpha=alpha)
            if alpha_shape is None:
                roi_outline = []
            elif hasattr(alpha_shape, 'geoms'):
                warnings.warn(f"Multi-part geometry for ROI '{roi}'; using largest polygon.")
                largest = max(alpha_shape.geoms, key=lambda g: g.area)
                roi_outline = [] if largest.is_empty else [[xv, yv] for xv, yv in largest.exterior.coords]
            else:
                roi_outline = [] if alpha_shape.is_empty else [[xv, yv] for xv, yv in alpha_shape.exterior.coords]
            outlines_roi[i] = roi_outline
    return outlines_roi


def disk_to_lab(r, theta, L=75):
    """
    Map a point in the unit disk (r, theta) to the CIE LAB color space with fixed L* and a 45° rotation.
    
    Parameters:
    r: float
        Radius in the unit disk (0 to 1)
    theta: float
        Angle in radians (0 to 2π)
    L: float
        Fixed L* value (lightness) in LAB color space (0 to 100)
    
    Returns:
    tuple: (L, a, b) coordinates in the CIE LAB color space
    """
    # Input validation
    if not (0 <= r <= 1):
        raise ValueError("r must be in the range [0, 1]")
    # Scale the radius to get an appropriate range for a* and b*
    # At typical L=75, the a* and b* ranges are roughly [-100, 100]
    max_radius = 100  # Maximum value for a* and b*
    
    # Convert polar coordinates to a* and b* coordinates
    a = r * max_radius * np.cos(theta + np.pi/4)
    b = r * max_radius * np.sin(theta + np.pi/4)
    
    return L, a, b

def lab_to_rgb(L, a, b):
    """
    Convert LAB color to sRGB for display.
    
    Parameters:
    L, a, b (float): CIE LAB color coordinates
    
    Returns:
    tuple: (R, G, B) values in range [0, 1], or None if the color is out of gamut
    """
    try:
        # Create a LAB color object
        lab = LabColor(L, a, b, illuminant="d50", observer="2")
        
        # Convert to XYZ
        xyz = convert_color(lab, XYZColor)
        
        # Convert to sRGB
        rgb = convert_color(xyz, sRGBColor)
        
        # Get RGB values
        rgb_values = (rgb.rgb_r, rgb.rgb_g, rgb.rgb_b)
        
        # Check if the color is in the sRGB gamut
        if (0 <= rgb.rgb_r <= 1) and (0 <= rgb.rgb_g <= 1) and (0 <= rgb.rgb_b <= 1):
            return rgb_values
        else:
            # Color is out of gamut
            print(f"LAB({L:.1f}, {a:.1f}, {b:.1f}) is out of sRGB gamut. Clamping to [0, 1].")
            return (rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
    except:
        # Exception during conversion (likely out of gamut)
        print(f"Exception converting LAB({L:.1f}, {a:.1f}, {b:.1f}) to sRGB.")
        return None
    

# from shapely.geometry import LineString
# from shapely.ops import nearest_points
def ray_intersection_with_boundary(pt, boundary_alpha):
    """
    Given a point (pt), compute the intersection of the ray from (0,0) to pt with the alpha shape boundary.
    Returns the intersection point as (x, y) or nearest point if no intersection.
    """
    ray = shapely.geometry.LineString([(0, 0), pt])
    boundary = boundary_alpha.boundary
    intersection = ray.intersection(boundary)
    if intersection.is_empty:
        print("No intersection, return nearest point instead.")
        # If no intersection, find the nearest point on the boundary to the ray
        nearest = shapely.ops.nearest_points(ray, boundary)
        return (nearest[1].x, nearest[1].y)
    # If multiple points, take the one closest to the origin (excluding the origin itself)
    if intersection.geom_type == 'MultiPoint':
        points = [p for p in intersection.geoms if not (p.x == 0 and p.y == 0)]
        if not points:
            return None
        # Sort by distance from origin
        points = sorted(points, key=lambda p: (p.x**2 + p.y**2))
        return (points[0].x, points[0].y)
    elif intersection.geom_type == 'Point':
        if intersection.x == 0 and intersection.y == 0:
            return None
        return (intersection.x, intersection.y)
    else:
        return None

def hex2cielab(df, boundary_alpha, L_val=65, mul_factor=50, radius_scale=1.1):
    """
    Compute color map based hex coord, using CIELAB color space.
    
    Parameters:
    -----------
    df_fit : pd.DataFrame
        DataFrame with columns 'h', 'v' for hexagonal coordinates
    boundary_alpha : shapely.geometry.Polygon
        Boundary polygon for ray intersection
    L_val : float, default=65
        Lightness value for LAB color space (0-100)
    mul_factor : float, default=50
        Multiplier to ensure intersection with boundary
    radius_scale : float, default=1.1
        Scale factor for radius calculation
        
    Returns:
    --------
    pd.DataFrame
        Input dataframe with added columns: 'r', 'theta', 'L', 'a', 'b', 'color_hex'
    """

    # check df has columns 'h' and 'v'
    if not all(col in df.columns for col in ['h', 'v']):
        raise ValueError("Input dataframe must contain 'h' and 'v' columns.")
    
    # Loop through each row to compute colors
    for i, row in df.iterrows():
        pt = (row['h'], row['v'])
        r = np.sqrt(pt[0]**2 + pt[1]**2)
        if r == 0:
            df.at[i, 'r'] = 0
            df.at[i, 'theta'] = 0
            continue
        mul = mul_factor / r  # ensure it intersects
        pt = (pt[0] * mul, pt[1] * mul)
        
        # Compute angle
        angle = np.arctan2(row['v'], row['h'])
        # Compute radius with intersection
        intersection = ray_intersection_with_boundary(pt, boundary_alpha)        
        radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2) * radius_scale
        
        # Save r and angle
        df.at[i, 'r'] = np.clip(radius, 0, 1)
        df.at[i, 'theta'] = angle
        
    # Convert disk coordinates to LAB
    r_vals = df['r'].values
    theta_vals = df['theta'].values
    L_vals = np.full_like(r_vals, L_val)
    lab = np.array([disk_to_lab(ri, ti, li) for ri, ti, li in zip(r_vals, theta_vals, L_vals)])
    df[['L', 'a', 'b']] = lab
    
    # Convert LAB to RGB
    lab = df[['L', 'a', 'b']].values
    _rgb = np.array([lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(lab[:, 0], lab[:, 1], lab[:, 2])])
    
    # Convert RGB to hex color strings
    _hex = [mpl.colors.to_hex(rgb) for rgb in _rgb]
    df['color_lab'] = _hex

    return df


def count_col(values, factor_thr):
    """
    Count the number of largest entries needed to reach a fraction (factor_thr) of
    the total sum, ignoring non-positive values.

    Accepts:
      - pandas.Series: values with an index (e.g., coordinate strings "x,y").
      - pandas.DataFrame with 1 column: treated as a Series.
      - numpy.ndarray / list-like 1D array: numeric values.

    Returns:
      - (count, kept_indices)
        * If input is a pandas.Series: kept_indices is a pandas Index of the
          original labels for the kept entries.
        * If input is array-like: kept_indices is a numpy array of integer
          positions (indices into the original array) for the kept entries.
    """
    if not (0 <= float(factor_thr) <= 1):
        raise ValueError("factor_thr must be in [0, 1]")

    # Normalize pandas DataFrame (single column) to Series
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("values DataFrame must have exactly one column")
        values = values.squeeze(axis=1)

    # Pandas Series path (preserve index labels)
    if isinstance(values, pd.Series):
        ser = pd.to_numeric(values, errors='coerce')
        ser = ser[ser > 0]
        if ser.empty:
            return 0, pd.Index([], dtype=values.index.dtype)

        sorted_values = ser.sort_values(ascending=False)
        cumsum = sorted_values.cumsum()
        total = cumsum.iloc[-1]
        threshold = float(factor_thr) * total
        mask_thr = cumsum <= threshold
        kept_indices = sorted_values[mask_thr].index
        return int(mask_thr.sum()), kept_indices

    # Numpy / list-like path (return integer positions)
    # arr = np.asarray(values)
    # if arr.ndim != 1:
    #     arr = arr.ravel()
    arr = np.asarray(values).ravel()
    # Filter positives, keep track of original positions
    pos_mask = arr > 0
    if not np.any(pos_mask):
        return 0, np.array([], dtype=int)
    pos_idx = np.nonzero(pos_mask)[0]
    arr_pos = arr[pos_mask] # positive values 
    # Sort descending and compute cumulative sum
    order = np.argsort(arr_pos)[::-1] # indices that would sort arr_pos in descending order
    arr_sorted = arr_pos[order]
    cumsum = np.cumsum(arr_sorted)
    total = cumsum[-1]
    threshold = float(factor_thr) * total
    kept_mask = cumsum <= threshold
    kept_pos_in_pos = order[kept_mask]
    kept_indices = pos_idx[kept_pos_in_pos]
    return int(kept_indices.size), kept_indices


def alpha_shape_2d(points, alpha=0.5, max_retries=6, alpha_reduction=0.1):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    
    Parameters:
    -----------
    points : array-like, shape (n, 2)
        Array of 2D points
    alpha : float
        Alpha value to control the shape's tightness.
        - Smaller alpha -> tighter fit (more concave)
        - Larger alpha -> looser fit (approaches convex hull)
        - alpha = 0 returns convex hull
    max_retries : int, default=5
        Maximum number of retries with reduced alpha
    alpha_reduction : float, default=0.1
        Factor to reduce alpha by on each retry
        
    Returns:
    --------
    edge_points : list of numpy arrays
        List of boundary components, each as (n, 2) array of (x, y) coordinates
    boundary_edges : set of tuples
        Set of edges (i, j) where i and j are indices in original points array
    areas : list of float
        List of areas corresponding to each boundary component in edge_points
    """

    points = np.array(points)
    
    if len(points) < 4:
        # Return convex hull for fewer than 4 points
        return points.tolist(), set(), [0.0]
    
    current_alpha = alpha
    
    for attempt in range(max_retries):
        # Compute Delaunay triangulation
        tri = Delaunay(points)
            
        # Find boundary edges (edges that appear in only one triangle)
        edge_count = defaultdict(int)
        for simplex in tri.simplices:
            pts = points[simplex]
            circumradius = compute_circumradius_triangle(pts)
            
            if current_alpha == 0 or circumradius < 1.0 / current_alpha:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                    edge_count[edge] += 1
        
        # Keep only boundary edges (those that appear once)
        boundary_edges = {edge for edge, count in edge_count.items() if count == 1}
        
        # Order the boundary points
        if boundary_edges:
            edge_points = order_boundaries(boundary_edges, points)
        else:
            edge_points = []

        # Check if we have multiple disconnected components
        if len(edge_points) > 1:
            if attempt < max_retries - 1:
                current_alpha -= alpha_reduction
                # warnings.warn(f"Multiple boundary components found. Retrying with alpha={current_alpha:.3f}")
                continue
            else:
                warnings.warn(f"Still have {len(edge_points)} components after {max_retries} retries")

        # Compute areas for each boundary component
        areas = []
        for boundary in edge_points:
            if len(boundary) >= 3:
                # Use shoelace formula for polygon area
                x = boundary[:, 0]
                y = boundary[:, 1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                areas.append(float(area))
            else:
                areas.append(0.0)
        
        # edge_points = pd.DataFrame(edge_points, columns=['h', 'v'])

        return edge_points, boundary_edges, areas
    
    return edge_points, boundary_edges, areas


def compute_circumradius_triangle(triangle_points):
    """
    Compute the circumradius of a triangle.
    
    Parameters:
    -----------
    triangle_points : array-like, shape (3, 2)
        The three vertices of the triangle
        
    Returns:
    --------
    radius : float
        The circumradius of the triangle
    """
    pts = np.array(triangle_points)
    a, b, c = pts[0], pts[1], pts[2]
    
    # Calculate side lengths
    a_len = np.linalg.norm(b - c)
    b_len = np.linalg.norm(a - c)
    c_len = np.linalg.norm(a - b)
    
    # Calculate area using cross product
    area = 0.5 * abs(np.cross(b - a, c - a))
    
    if area == 0:
        return float('inf')
    
    # Circumradius formula: R = (abc) / (4 * Area)
    circumradius = (a_len * b_len * c_len) / (4.0 * area)
    
    return circumradius
    
def order_boundaries(edges, points):
    """Find and order all disconnected boundary components."""
    if not edges:
        return []
    
    visited_edges = set()
    visited_nodes = set()
    all_boundaries = []
    all_boundaries_idx = []

    # Find all connected components
    for start_edge in edges:
        if start_edge in visited_edges:
            continue
            
        # Trace this component
        boundary = []
        current = start_edge[0]
        prev = None
        start = current
        
        while True:
            # find all edges that contain current node and a node that's not in visited_nodes
            current_edges = [e for e in edges if 
                            current in e and
                            not (set(e) <= visited_nodes) and
                            e not in visited_edges]
            
            # pick the first in current_edges, get the node that's not current
            if not current_edges:
                break
            
            # check current_edges
            current_edge = current_edges[0]
            neighbor = current_edge[0] if current_edge[1] == current else current_edge[1]
            
            # update
            prev = current
            current = neighbor
            boundary.append(current)
            visited_edges.add(current_edge)
            visited_nodes.add(current)
            
        # add the start node as the first element of the boundary
        boundary.insert(0, start)
        visited_nodes.add(start)
        ordered_points = [(points[i][0], points[i][1]) for i in boundary]


        all_boundaries.append(np.array(ordered_points))
        all_boundaries_idx.append(boundary)
    
    return all_boundaries

    
def count_col_ahull(df, thr_mode ="max", remove_frac = 0.1, alpha=0.5, max_retries=6, alpha_reduction=0.1):
    """

    """
    if not (0 <= float(remove_frac) <= 1):
        raise ValueError("remove_frac must be in [0, 1]")

    # check df is a single column DataFrame or Series
    if not isinstance(df, (pd.DataFrame, pd.Series)) or (isinstance(df, pd.DataFrame) and df.shape[1] != 1):
        raise ValueError("df must be a pandas DataFrame with a single column or a pandas Series")
    
    # check df is not empty
    if df.values.sum() == 0:
        warnings.warn("weights sum to 0 Returning NaN results.")
        return [], set(), [np.nan], (np.nan, np.nan) , None 

    # convert df to series if it's a single column DataFrame
    if isinstance(df, pd.DataFrame):
        df = df.squeeze(axis=1)
        
    # get xy coord from index
    xy_coord = df.index.to_series().str.split(',', expand=True).astype(float)
    xy_coord.columns = ['x', 'y']

    # # check df has at least 3 positive entries for alpha shape
    # if (df.values > 0).sum() < 3:
    #     warnings.warn("Not enough positive entries for alpha shape. Returning column count directly.")
    #     return [], set(), [(df.values > 0).sum()], (np.nan, np.nan) , None      


    # thresholding effective weight points for alpha shape, "max" or "cumsum"  
    if thr_mode == "max":
        # remove points with effective weight smaller than x% of max
        thr_val = remove_frac * df.values.max()
    elif thr_mode == "cumsum":
        # remove smallest x% of total weight
        weights_sorted = df.sort_values()
        cutoff = remove_frac * weights_sorted.sum()
        thr_val = weights_sorted[weights_sorted.cumsum() >= cutoff].iloc[0]
    else:
        raise ValueError("Invalid thr_mode. Choose 'max' or 'cumsum'.")
    # ## ## thresholding by max
    # thr_val = thr_max * df.max()
    # mask_pos = df.values > thr_val
    # ## ## thresholding, order and compute cumsum, keep the largest until reaching thr_max of total
    # # df_sorted = df.sort_values(ascending=False)
    # # cumsum = df_sorted.cumsum()
    # # total = df.sum()
    # # threshold = (1 - thr_max) * total
    # # mask_pos = cumsum <= threshold
    
    # thresholding
    mask_pos = df.values > thr_val
    xy_coord_kept = xy_coord[mask_pos]
    df = df[mask_pos]

    # Check if we have enough non-collinear points
    if len(xy_coord_kept) == 0:
        return None, None, [0], (np.nan, np.nan), None
    
    # compute com weighted by values
    com_x = np.average(xy_coord_kept['x'], weights=df.values)
    com_y = np.average(xy_coord_kept['y'], weights=df.values)

    if len(xy_coord_kept) < 3:
        return None, None, [xy_coord_kept.shape[0]], (com_x, com_y), xy_coord_kept
    
    # Check for collinearity by computing convex hull area
    try:
        hull = ConvexHull(xy_coord_kept.values)
        if hull.volume < 1e-10:  # Nearly zero area
            return None, None, [xy_coord_kept.shape[0]], (com_x, com_y), None
    except:
        return None, None, [xy_coord_kept.shape[0]], (com_x, com_y), None
    
    # df.index is a square grid, with range x ~ [-15 19], range y ~ [7 72]
    # nb points in square grid is [+1 +1], in hex grid of side = 1 is [1+1/2, 3/2/sqrt(3)]
    # to morph into a regular hex grid with side length = 1
    # multiply x by 3/2 and y by sqrt(3)/2
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] * 3/2
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] * np.sqrt(3)/2

    # correction for edge points
    # for each xy_coord_kept, add six points around it with distance 1 in the corrected hex grid
    directions = np.array([[1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    aux_points = []
    for _, row in xy_coord_kept.iterrows():
        for d in directions:
            aux_points.append([row['x'] + d[0], row['y'] + d[1]])
    # combine edge points with xy_coord_kept for alpha shape
    aux_points = np.array(aux_points)
    aux_points = np.vstack([xy_coord_kept.values, aux_points])

    edge_points, edges, area = alpha_shape_2d(aux_points, alpha=alpha, max_retries=max_retries, alpha_reduction=alpha_reduction)

    if len(area) != 1:
        area = [np.nan]
    
    # in units of hex cell area, which is 3*sqrt(3)/2 * (side length)^2. 
    # For side length = 1 a in regular hex grid:
    area = [a / (3 * np.sqrt(3) / 2) for a in area]

    # go back to the original coord
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] / (3/2)
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] / (np.sqrt(3)/2)
    if len(edge_points) == 1:
        edge_points[0][:, 0] = edge_points[0][:, 0] / (3/2)
        edge_points[0][:, 1] = edge_points[0][:, 1] / (np.sqrt(3)/2)
    else:
        warnings.warn("Multiple ahulls.")

    return edge_points, edges, area, (com_x, com_y), xy_coord_kept


def participation_ratio(df):
    """
    Compute the participation ratio (1/Σpᵢ² , effective number of columns) from a
    weight distribution over hex grid positions.

    PR = (sum w_i)^2 / sum(w_i^2)

    Only positive weights are included. The result can be converted to a
    spatial area by multiplying by the area of one hex cell.

    Parameters
    ----------
    df : pd.Series or single-column pd.DataFrame
        Weights indexed by 'h,v' coordinate strings (same format as
        effwt_visr columns).

    Returns
    -------
    pr : float
        Participation ratio (effective number of columns).
        Returns np.nan if all weights are zero or negative.
    """
    w = np.asarray(df.values, dtype=float).ravel()
    w = w[w > 0]

    if len(w) == 0:
        return np.nan

    sum_w = w.sum()
    sum_w2 = (w ** 2).sum()

    return sum_w ** 2 / sum_w2


def assign_vpn_hop(meta_cb_vpn,  rois_cb, oltypes, rf_fit, mode='vic', thr_vic=0, thr_inwt=10):
    """
    Compute hopping groups (ghop) for visual input connectivity starting from VPNs
    
    Parameters:
    -----------
    meta_cb_vpn : pd.DataFrame
        The meta DataFrame containing ['bodyId', 'cell_type_side', 'instance', 'vision', 'ht', 'main_groups']
    rois_cb : list
        List of central brain ROIs
    oltypes : pd.DataFrame
        DataFrame containing all optic lobe cell types
    rf_fit : pd.DataFrame
        DataFrame containing RF fit data with bodyId and r2 columns
    mode : str, optional
        Either 'vic' (visual input contribution) or 'res' (resolution) (default: 'vic')
    thr_vic : float, optional
        Visual input contribution threshold (default: 0)
    thr_inwt : float, optional
        Input weight threshold percentage (default: 10)
    
    Returns:
    --------
    meta_cb_vpn : pd.DataFrame
        Updated meta DataFrame with 'ghop' column for each bodyId
    meta_type_cb_vpn : pd.DataFrame
        Updated meta DataFrame with 'ghop' column for each cell type and instance
    """
            
    # Meta by type - different aggregations based on mode
    if mode == 'vic':
        meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
            bodyId_count=('bodyId', 'count'),
            vision=('vision', 'median'),
            ht=('ht', 'median'),
            main_groups=('main_groups', 'first'),
        ).reset_index()
        
        # Filter T0 instances (VIC mode)
        inst_t0 = meta_type_cb_vpn[
            (meta_type_cb_vpn['vision'] > thr_vic/100) & 
            (meta_type_cb_vpn['main_groups'] == 'VPN')
        ]['instance'].values
        
    elif mode == 'res':
        meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
            bodyId_count=('bodyId', 'count'),
            area_fit=('area_fit', 'median'),
            area_fit_input=('area_fit_input', 'median'),
            r2=('r2', 'median'),
            col_count=('col_count', 'median'),
            col_count_input=('col_count_input', 'median'),
            vision=('vision', 'median'),
            vision_cv=('vision', lambda x: round(np.std(x) / np.mean(x), 2) if np.mean(x) != 0 else np.nan),
            ht=('ht', 'median'),
            main_groups=('main_groups', 'first'),
        ).reset_index()
        
        # Filter T0 instances (resolution mode)
        inst_t0 = meta_type_cb_vpn[
            (meta_type_cb_vpn['vision'] > thr_vic/100) &
            (meta_type_cb_vpn['r2'] >= 0) &
            (meta_type_cb_vpn['main_groups'] == 'VPN')
        ]['instance'].values
    else:
        raise ValueError(f"mode must be 'vic' or 'res', got '{mode}'")

    # Get T0 -> T1 connections
    neuron_df, connection_df = fetch_adjacencies(
        sources=inst_t0,
        targets=None,
        rois=rois_cb,
        min_total_weight=1
    )
    conn_t0t1 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
        .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
        .agg({'weight': 'sum'})\
        .reset_index()\
        .sort_values(by='weight', ascending=False)

    # Filter T1 candidates
    conn_t0t1 = conn_t0t1[
        ~conn_t0t1['instance_post'].str.contains("\\'") & 
        ~conn_t0t1['instance_post'].str.contains("\\?") &
        ~conn_t0t1['instance_post'].str.contains("\\+") &
        ~conn_t0t1['instance_post'].str.contains("unclear") &
        ~conn_t0t1['instance_post'].isin(oltypes['instance'])
    ]
    inst = conn_t0t1['instance_post'].unique()

    # Calculate input weights for T1
    neuron_df, connection_df = fetch_adjacencies(sources=None, targets=inst, min_total_weight=1)
    idwt_t1 = neuprint.merge_neuron_properties(
        neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

    df = pd.merge(
        conn_t0t1.groupby(['bodyId_post', 'instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
        idwt_t1[['bodyId_post', 'weight']], on='bodyId_post', how='left'
    )
    df['inwt'] = df['conn_wt'] / df['weight']
    df = pd.merge(df, rf_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')
    
    df_inst = df.groupby(['instance_post']).agg(
        bodyId_count=('bodyId_post', 'count'),
        inwt_median=('inwt', 'median'),
        r2=('r2', 'median')
    ).reset_index()

    # Filter T2 instances - different criteria for VIC vs resolution
    if mode == 'vic':
        inst_t1 = df_inst.loc[
            df_inst['inwt_median'] >= thr_inwt/100, 
            'instance_post'
        ].unique()
    elif mode == 'res':
        inst_t1 = df_inst.loc[
            (df_inst['inwt_median'] >= thr_inwt/100) & 
            (df_inst['r2'] >= 0), 
            'instance_post'
        ].unique()

    # Get T0/T1 -> T2 connections
    neuron_df, connection_df = fetch_adjacencies(
        sources=np.concatenate([inst_t0, inst_t1]),
        targets=None,
        rois=rois_cb,
        min_total_weight=1
    )
    conn_t0t1t2 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
        .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
        .agg({'weight': 'sum'})\
        .reset_index()\
        .sort_values(by='weight', ascending=False)

    # Filter T2 candidates 
    conn_t0t1t2 = conn_t0t1t2[
        ~conn_t0t1t2['instance_post'].str.contains("\\'") & 
        ~conn_t0t1t2['instance_post'].str.contains("\\?") &
        ~conn_t0t1t2['instance_post'].str.contains("\\+") &
        ~conn_t0t1t2['instance_post'].str.contains("unclear") &
        ~conn_t0t1t2['instance_post'].isin(oltypes['instance']) &
        ~conn_t0t1t2['instance_post'].isin(inst_t1)
    ]
    inst = conn_t0t1t2['instance_post'].unique()

    # Calculate input weights for T2
    neuron_df, connection_df = fetch_adjacencies(sources=None, targets=inst, min_total_weight=1)
    idwt = neuprint.merge_neuron_properties(
        neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

    df = pd.merge(
        conn_t0t1t2.groupby(['bodyId_post', 'instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
        idwt[['bodyId_post', 'weight']], on='bodyId_post', how='left'
    )
    df['inwt'] = df['conn_wt'] / df['weight']
    df = pd.merge(df, rf_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')
    
    df_inst = df.groupby(['instance_post']).agg(
        bodyId_count=('bodyId_post', 'count'),
        inwt_median=('inwt', 'median'),
        r2=('r2', 'median')
    ).reset_index()

    # Filter T2 instances - different criteria for VIC vs resolution
    if mode == 'vic':
        inst_t2 = df_inst.loc[
            df_inst['inwt_median'] >= thr_inwt/100, 
            'instance_post'
        ].unique()
    elif mode == 'res':
        inst_t2 = df_inst.loc[
            (df_inst['inwt_median'] >= thr_inwt/100) & 
            (df_inst['r2'] >= 0), 
            'instance_post'
        ].unique()

    # Assign hopping groups
    meta_cb_vpn['ghop'] = meta_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

    meta_type_cb_vpn['ghop'] = meta_type_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'
    
    return meta_cb_vpn, meta_type_cb_vpn


def _rotate_outlines(outlines, rot_mat, center):
    if isinstance(outlines[0], pd.DataFrame):
        # store the column name to recover later
        col_names = outlines[0].columns
    rotated = []
    for poly in outlines:
        pts = np.asarray(poly, dtype=float)
        if pts.size == 0:
            rotated.append([])
            continue
        pts_rot = (rot_mat @ (pts - center).T).T + center
        if col_names is not None:
            rotated.append(pd.DataFrame(pts_rot, columns=col_names))
        else:
            rotated.append(pts_rot.tolist())
    return rotated
    
def plot_extreme_projection_xy(
    df: pd.DataFrame,
    outlines_bkgd: list,
    xrange: list | tuple = [0, 104000],
    yrange: list | tuple = [4000, 56000],
    *,
    rotation_deg: float = 3.0,
    xnbins: int = 208, 
    ynbins: int = 104,
    agg: str = 'largest',
    cmap_div: bool = False,
    agg_frac: float = 0.1,
    agg_num: float = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    im_norm: str ='linear',
    clip: bool = True,
    colorbar_label: str = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, mpl.image.AxesImage]:
    """Plot a synapse-value heatmap with ROI outlines."""

    def _grid_stat(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size <= agg_num:
            return np.nan
        n_keep = max(1, int(np.ceil(agg_frac * arr.size)))
        arr_sorted = np.sort(arr)
        sel = arr_sorted[-n_keep:] if agg == 'largest' else arr_sorted[:n_keep]
        return float(np.mean(sel))

    # check df has required columns, x ,y, val
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    required_cols = {'x', 'y', 'val'}
    if not all(col in df.columns for col in ['x', 'y', 'val']):
        raise ValueError(f"df must contain columns 'x', 'y', and 'val'.")
    df = df[['x', 'y', 'val']].dropna().copy()
    df = df.astype({'x': float, 'y': float}, copy=False)
    if df.empty:
        print("No data available after filtering for heatmap plotting.")
        return None, None, None
    
    if agg not in ['largest', 'smallest']:
        raise ValueError("agg must be either 'largest' or 'smallest'.")

    # rotate [x y]
    if rotation_deg:
        center = np.array([(xrange[0] + xrange[1]) / 2, (yrange[0] + yrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords = df[['x', 'y']].to_numpy()
        df.loc[:, ['x', 'y']] = ((rot_mat @ (coords - center).T).T + center)
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)

    x_bins = np.linspace(xrange[0], xrange[1], xnbins + 1)
    y_bins = np.linspace(yrange[0], yrange[1], ynbins + 1)
    in_bounds = (
        (df['x'] >= x_bins[0]) & (df['x'] <= x_bins[-1]) &
        (df['y'] >= y_bins[0]) & (df['y'] <= y_bins[-1])
    )
    df = df.loc[in_bounds].copy()
    if df.empty:
        raise ValueError('No data falls inside the requested plotting bounds.')

    x_idx = np.digitize(df['x'], x_bins) - 1
    y_idx = np.digitize(df['y'], y_bins) - 1
    df = df.assign(x_grid=x_idx, y_grid=y_idx, val=df['val'])

    grid_stats = (
        df.groupby(['x_grid', 'y_grid'])['val']
        .agg(_grid_stat)
        .reset_index()
    )

    heatmap_data = np.full((ynbins, xnbins), np.nan, dtype=float)
    for _, row in grid_stats.iterrows():
        heatmap_data[int(row['y_grid']), int(row['x_grid'])] = row['val']

    if vmin is None or vmax is None:
        finite_vals = heatmap_data[np.isfinite(heatmap_data)]
        if finite_vals.size == 0:
            raise ValueError('Heatmap contains only NaNs; cannot infer color limits.')
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 5))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 95))

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    if agg == 'largest':
        if cmap_div:
            cmap_obj = mpl.colormaps.get_cmap('RdYlBu_r')
        else:
            cmap_obj = mpl.colormaps.get_cmap('plasma')
    elif agg == 'smallest':
        if cmap_div:
            cmap_obj = mpl.colormaps.get_cmap('RdYlBu_r')
        else:
            cmap_obj = mpl.colormaps.get_cmap('plasma_r')
    cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, 'copy') else cmap_obj
    cmap_obj.set_bad(color='black')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        heatmap_data,
        extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
        origin='lower',
        aspect='equal',
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=im_norm,
        interpolation='none'
    )
    # Force a compact raster in vector exports (PDF/SVG). Without this, a clipped
    # imshow with many filled bins makes matplotlib's PDF backend fall back to
    # ~10k+ vector line segments per panel (dense heatmaps blow up to ~17 kB each);
    # rasterizing keeps it one small image while colorbar/outline/text stay vector.
    im.set_rasterized(True)

    if clip:
        pts = np.asarray(outlines_bkgd[0], dtype=float)
        clip_patch = Polygon(pts, transform=ax.transData, facecolor='none')
        im.set_clip_path(clip_patch)
        pts = np.asarray(outlines_bkgd[1], dtype=float)
        clip_patch = Polygon(pts, transform=ax.transData, facecolor='white')
        ax.add_patch(clip_patch)
        outline_color = 'black'
    else:
        outline_color = 'white'

    # add outlines
    for poly in outlines_bkgd:
        pts = np.asarray(poly, dtype=float)
        if pts.size >= 4:
            ax.plot(pts[:, 0], pts[:, 1], color=outline_color, linewidth=1)

    # add scale bar, 8nm, 10um scale bar
    ax.plot([60000, 60000+12500], [42000, 42000], color='black', linewidth=2)
    ax.text(66250, 43000, '100 µm', ha='center', va='top', fontsize=10, color='black')

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.invert_yaxis()

    # remove axes
    ax.axis('off')

    # set colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.75)
    if colorbar_label is not None:
        cbar.ax.set_xlabel(colorbar_label, rotation=0, ha='center')
        cbar.ax.xaxis.set_label_position('bottom')
    
    return fig, ax, im


# todo, combine with above
def plot_extreme_projection_xz(
    df: pd.DataFrame,
    outlines_bkgd: list,
    xrange: list | tuple = [26000, 70000],
    zrange: list | tuple = [50000, 138000],
    *,
    rotation_deg: float = 0.0,
    xnbins: int = 208,
    znbins: int = 416,
    agg: str = 'largest',
    agg_frac: float = 0.1,
    vmin: float | None = None,
    vmax: float | None = None,
    im_norm: str ='linear',
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, mpl.image.AxesImage]:
    """Plot a synapse-value heatmap with ROI outlines."""

    def _grid_stat(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        n_keep = max(1, int(np.ceil(agg_frac * arr.size)))
        arr_sorted = np.sort(arr)
        sel = arr_sorted[-n_keep:] if agg == 'largest' else arr_sorted[:n_keep]
        return float(np.mean(sel))

    # check df has required columns, x ,y, val
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    required_cols = ['x', 'z', 'val']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df must contain columns {required_cols}.")
    df = df[required_cols].dropna().copy()
    df = df.astype({'x': float, 'z': float}, copy=False)
    # if no data after filtering, return a msg and stop the function
    if df.empty:
        print("No data available after filtering for heatmap plotting.")
        return None, None, None
    
    if agg not in ['largest', 'smallest']:
        raise ValueError("agg must be either 'largest' or 'smallest'.")

    # rotate [x z]
    if rotation_deg != 0.0:
        center = np.array([(xrange[0] + xrange[1]) / 2, (zrange[0] + zrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords = df[['x', 'z']].to_numpy()
        df.loc[:, ['x', 'z']] = ((rot_mat @ (coords - center).T).T + center)
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)

    x_bins = np.linspace(xrange[0], xrange[1], xnbins + 1)
    z_bins = np.linspace(zrange[0], zrange[1], znbins + 1)
    in_bounds = (
        (df['x'] >= x_bins[0]) & (df['x'] <= x_bins[-1]) &
        (df['z'] >= z_bins[0]) & (df['z'] <= z_bins[-1])
    )
    df = df.loc[in_bounds].copy()
    if df.empty:
        warnings.warn('No data falls inside the requested plotting bounds.')
        return None, None, None

    x_idx = np.digitize(df['x'], x_bins) - 1
    z_idx = np.digitize(df['z'], z_bins) - 1
    df = df.assign(x_grid=x_idx, z_grid=z_idx, val=df['val'])

    grid_stats = (
        df.groupby(['x_grid', 'z_grid'])['val']
        .agg(_grid_stat)
        .reset_index()
    )

    heatmap_data = np.full((znbins, xnbins), np.nan, dtype=float)
    for _, row in grid_stats.iterrows():
        heatmap_data[int(row['z_grid']), int(row['x_grid'])] = row['val']

    if vmin is None or vmax is None:
        finite_vals = heatmap_data[np.isfinite(heatmap_data)]
        if finite_vals.size == 0:
            raise ValueError('Heatmap contains only NaNs; cannot infer color limits.')
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 5))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 95))

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    if agg == 'largest':
        cmap_obj = mpl.colormaps.get_cmap('plasma')
    else:
        cmap_obj = mpl.colormaps.get_cmap('plasma_r')
    cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, 'copy') else cmap_obj
    cmap_obj.set_bad(color='black')

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 8))
    else:
        fig = ax.figure

    im = ax.imshow(
        heatmap_data,
        extent=[x_centers[0], x_centers[-1], z_centers[0], z_centers[-1]],
        origin='lower',
        aspect='equal',
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=im_norm,
        interpolation='none'
    )
    # Force a compact raster in vector exports (PDF/SVG). Without this, a clipped
    # imshow with many filled bins makes matplotlib's PDF backend fall back to
    # ~10k+ vector line segments per panel (dense heatmaps blow up to ~17 kB each);
    # rasterizing keeps it one small image while colorbar/outline/text stay vector.
    im.set_rasterized(True)

    for poly in outlines_bkgd:
        pts = np.asarray(poly, dtype=float)
        if pts.size >= 4:
            ax.plot(pts[:, 0], pts[:, 1], color='white', linewidth=1)

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(zrange[0], zrange[1])
    ax.invert_yaxis()

    # if ax is not None and ax.figure is fig:
    fig.colorbar(im, ax=ax, orientation='vertical')
    
    return fig, ax, im


def plot_neuron_with_outlines(n_ls, outlines_bkgd,     
                            neuron_color='black', 
                            outline_color='gray',
                            xrange: list | tuple = [20000, 80000],
                            yrange: list | tuple = [000, 50000],
                            rotation_deg: float = 3.0,
                            width=360,
                            height=300,
                            show_legend=False):
    """
    Create a 3D plotly figure with neuron(s) and background outlines.
    
    Parameters
    ----------
    n_ls : list of navis neuron mesh/skeleton
        List of neurons (mesh or skeleton) to plot
    outlines_bkgd : list of pd.DataFrame
        List of dataframes containing outline coordinates with columns 'x1', 'y1', and optionally 'z'
    neuron_color : str or list of str, optional
        Color(s) for neuron(s). If n is a list and neuron_color is a single color, 
        all neurons will have the same color. Default: 'black'
    outline_color : str, optional
        Color for outline traces. Default: 'gray'
    xrange : list or tuple, optional
        X-axis range for rotation center. Default: [0, 104000]
    yrange : list or tuple, optional
        Y-axis range for rotation center. Default: [4000, 56000]
    rotation_deg : float, optional
        Rotation angle in degrees to apply to the neuron and outlines around the Z-axis. Default: 3.0
    width : int, optional
        Width of the figure in pixels. Default: 400
    height : int, optional
        Height of the figure in pixels. Default: 400
    show_legend : bool, optional
        Whether to show legend in the figure. Default: False
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure with neuron(s) and outlines
    """
    fig = go.Figure()
    
    # Handle single neuron vs list of neurons
    if not isinstance(n_ls, navis.core.neuronlist.NeuronList):
        n_ls = [n_ls]

    # Handle neuron colors
    if isinstance(neuron_color, str):
        neuron_colors = [neuron_color] * len(n_ls)
    elif isinstance(neuron_color, list):
        if len(neuron_color) != len(n_ls):
            raise ValueError(f"Length of neuron_color ({len(neuron_color)}) must match number of neurons ({len(n_ls)})")
        neuron_colors = neuron_color
    else:
        raise ValueError("neuron_color must be a string or list of strings")
    
    # rotate around z-axis
    if rotation_deg:
        center = np.array([(xrange[0] + xrange[1]) / 2, (yrange[0] + yrange[1]) / 2])
        theta = np.radians(rotation_deg)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)
        if isinstance(neuron_colors[0], navis.MeshNeuron):
            for i in range(len(n_ls)):
                coords = n_ls[i].vertices
                n_ls[i].vertices = ((rot_mat @ (coords - center).T).T + center)
        elif isinstance(neuron_colors[0], navis.TreeNeuron):
            for i in range(len(n_ls)):
                coords = n_ls[i].nodes[['x','y','z']]
                n_ls[i].skeleton = ((rot_mat @ (coords - center).T).T + center)

    # Add neuron traces
    fig_n = navis.plot3d(n_ls, inline=False, color=neuron_color, backend='plotly')
    for trace in fig_n.data:
        fig.add_trace(trace)
    
    # fig = go.Figure(data= fig_n.data)

    # Add outline traces
    for outline in outlines_bkgd:
        if len(outline) > 0:
            fig.add_trace(go.Scatter3d(
                x=outline['x1'],
                y=outline['y1'],
                z=outline['z'] if 'z' in outline.columns else [30000] * len(outline),
                mode='lines',
                line=dict(color=outline_color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        scene=dict(
            camera=dict(
                projection=dict(type='orthographic'),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z= -1),
                up=dict(x=0, y=-1, z=0)
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor="rgba(0, 0, 0, 0)"
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=show_legend
    )
    
    return fig