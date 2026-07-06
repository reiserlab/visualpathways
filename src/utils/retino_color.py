"""
Retinotopic color mapping: map hex / unit-disk coordinates to display colors.

Converts positions on the unit disk to the CIE LAB color space (fixed
lightness, 45 degree rotation), converts LAB to sRGB for display, and maps a
hex-coordinate dataframe to per-column hex color strings via a ray/boundary
radius normalization. Used to build retinotopic color wheels for the optic
lobe. Depends on colormath for color-space conversion and on
:func:`utils.alpha_shape.ray_intersection_with_boundary` for the disk radius.
"""

import matplotlib as mpl
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, XYZColor, sRGBColor

from utils.alpha_shape import ray_intersection_with_boundary


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
