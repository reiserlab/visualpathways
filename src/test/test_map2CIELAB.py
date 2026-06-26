import numpy as np
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def disk_to_lab(r, theta, L=75):
    """
    Map a point in the unit disk (r, theta) to the CIE LAB color space with fixed L* and a 45° rotation.
    
    Parameters:
    r (float): Radius in the unit disk (0 to 1)
    theta (float): Angle in radians (0 to 2π)
    L (float): Fixed L* value (lightness) in LAB color space (0 to 100)
    
    Returns:
    tuple: (L, a, b) coordinates in the CIE LAB color space
    """
    # Input validation
    if not (0 <= r <= 1):
        raise ValueError("r must be in the range [0, 1]")
    if not (0 <= theta <= 2 * np.pi):
        raise ValueError("theta must be in the range [0, 2π]")
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

def create_lab_color_wheel(L=75, resolution=100):
    """
    Create and display a color wheel mapping disk positions to LAB colors.
    
    Parameters:
    L (float): Fixed L* value (lightness) in LAB color space (0 to 100)
    resolution (int): Number of points along each dimension
    
    Returns:
    matplotlib Figure: The generated figure
    """
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Setup the unit disk in a*b* space
    ax1.set_aspect('equal')
    ax1.set_xlim(-110, 110)
    ax1.set_ylim(-110, 110)
    
    # Draw a background circle for the a*b* plane
    background_circle = Circle((0, 0), 100, fill=False, color='black')
    ax1.add_patch(background_circle)
    
    # Add coordinate axes
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'CIE LAB Color Space (L*={L})')
    ax1.set_xlabel('a* (green to red)')
    ax1.set_ylabel('b* (blue to yellow)')
    
    # Setup the unit disk for input
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    
    # Draw a unit circle
    unit_circle = Circle((0, 0), 1, fill=False, color='black')
    ax2.add_patch(unit_circle)
    
    # Add coordinate axes
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Unit Disk Input Space')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Create a grid of points in the unit disk
    r_values = np.linspace(0, 1, resolution)
    theta_values = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_values, theta_values)
    
    # Calculate the x, y positions in the unit disk
    x_disk = r_grid * np.cos(theta_grid)
    y_disk = r_grid * np.sin(theta_grid)
    
    # Initialize arrays for storing colorful points
    valid_x_disk = []
    valid_y_disk = []
    valid_colors = []
    valid_a = []
    valid_b = []
    
    # Create a color visualization
    for i in range(resolution):
        for j in range(resolution):
            r = r_grid[i, j]
            theta = theta_grid[i, j]
            
            # Disk coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Map to LAB coordinates
            lab_L, lab_a, lab_b = disk_to_lab(r, theta, L)
            
            # Try to convert to RGB
            rgb_color = lab_to_rgb(lab_L, lab_a, lab_b)
            
            # If a valid RGB color is obtained, store the values
            if rgb_color is not None:
                valid_x_disk.append(x)
                valid_y_disk.append(y)
                valid_colors.append(rgb_color)
                valid_a.append(lab_a)
                valid_b.append(lab_b)
    
    # Plot colors in both spaces
    # Plot valid LAB colors in the a*b* plane
    scatter1 = ax1.scatter(valid_a, valid_b, c=valid_colors, s=15)
    
    # Plot valid colors in the unit disk
    scatter2 = ax2.scatter(valid_x_disk, valid_y_disk, c=valid_colors, s=15)
    
    # Draw reference points at 45° angles in both spaces
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        # Unit disk reference
        x_ref = 0.8 * np.cos(angle)
        y_ref = 0.8 * np.sin(angle)
        ax2.plot(x_ref, y_ref, 'o', color='black', markersize=8)
        
        # LAB space reference
        lab_L, lab_a, lab_b = disk_to_lab(0.8, angle, L)
        ax1.plot(lab_a, lab_b, 'o', color='black', markersize=8)
        
        # Connect reference points with lines
        if angles[0] == angle:  # Only annotate the first point
            ax2.annotate(f"{angle:.1f} rad", (x_ref, y_ref), 
                        xytext=(x_ref*1.1, y_ref*1.1), fontsize=8)
            ax1.annotate(f"{angle:.1f} rad", (lab_a, lab_b), 
                        xytext=(lab_a*1.1, lab_b*1.1), fontsize=8)
    
    # Add color labels
    ax1.text(0, 90, "YELLOW", ha='center', fontsize=10)
    ax1.text(90, 0, "RED", ha='center', fontsize=10)
    ax1.text(0, -90, "BLUE", ha='center', fontsize=10)
    ax1.text(-90, 0, "GREEN", ha='center', fontsize=10)
    
    # Add sRGB gamut indication
    ax1.text(0, -110, "Note: White areas are outside sRGB gamut", 
             ha='center', fontsize=9, style='italic')
    
    # Add explanation
    ax2.text(0, -1.1, "Angle → Hue, Radius → Chroma", 
             ha='center', fontsize=10, style='italic')
    
    plt.suptitle(f'Mapping from Unit Disk to CIE LAB Color Space (L*={L})', fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_lab_slices(L_values=[25, 50, 75, 90], resolution=50):
    """
    Visualize multiple L* slices of the LAB color space from disk mappings.
    
    Parameters:
    L_values (list): List of L* values to visualize
    resolution (int): Number of points along each dimension
    """
    # Create a figure with subplots for each L value
    fig, axes = plt.subplots(1, len(L_values), figsize=(16, 5))
    
    for i, L in enumerate(L_values):
        ax = axes[i]
        ax.set_aspect('equal')
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        
        # Draw a background circle
        background_circle = Circle((0, 0), 100, fill=False, color='black')
        ax.add_patch(background_circle)
        
        # Add coordinate axes
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'L* = {L}')
        
        if i == 0:
            ax.set_ylabel('b* (blue to yellow)')
        if i == len(L_values) // 2:
            ax.set_xlabel('a* (green to red)')
        
        # Create a grid of points
        r_values = np.linspace(0, 1, resolution)
        theta_values = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        r_grid, theta_grid = np.meshgrid(r_values, theta_values)
        
        # Initialize arrays for storing points
        valid_colors = []
        valid_a = []
        valid_b = []
        
        # Create a color visualization
        for j in range(resolution):
            for k in range(resolution):
                r = r_grid[j, k]
                theta = theta_grid[j, k]
                
                # Map to LAB coordinates
                lab_L, lab_a, lab_b = disk_to_lab(r, theta, L)
                
                # Try to convert to RGB
                rgb_color = lab_to_rgb(lab_L, lab_a, lab_b)
                
                # If a valid RGB color is obtained, store the values
                if rgb_color is not None:
                    valid_colors.append(rgb_color)
                    valid_a.append(lab_a)
                    valid_b.append(lab_b)
        
        # Plot valid LAB colors
        ax.scatter(valid_a, valid_b, c=valid_colors, s=15)
    
    plt.suptitle('CIE LAB Color Space at Different L* Values', fontsize=16)
    plt.tight_layout()
    
    return fig

def calculate_lab_gamut_coverage(L=75, resolution=100):
    """
    Calculate and visualize the sRGB gamut coverage in LAB space.
    
    Parameters:
    L (float): L* value to analyze
    resolution (int): Grid resolution
    
    Returns:
    tuple: (coverage percentage, figure)
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    
    # Create a grid covering the a*b* plane
    a_values = np.linspace(-100, 100, resolution)
    b_values = np.linspace(-100, 100, resolution)
    a_grid, b_grid = np.meshgrid(a_values, b_values)
    
    # Calculate distance from origin
    distance = np.sqrt(a_grid**2 + b_grid**2)
    
    # Initialize an array to store which points are inside the sRGB gamut
    in_gamut = np.zeros((resolution, resolution), dtype=bool)
    
    # Check each point
    total_points = 0
    valid_points = 0
    valid_a = []
    valid_b = []
    valid_colors = []
    
    for i in range(resolution):
        for j in range(resolution):
            a = a_grid[i, j]
            b = b_grid[i, j]
            
            # Only check points within the circle of radius 100
            if distance[i, j] <= 100:
                total_points += 1
                
                # Check if this LAB color is in the sRGB gamut
                rgb_color = lab_to_rgb(L, a, b)
                
                if rgb_color is not None:
                    in_gamut[i, j] = True
                    valid_points += 1
                    valid_a.append(a)
                    valid_b.append(b)
                    valid_colors.append(rgb_color)
    
    # Calculate coverage percentage
    coverage = (valid_points / total_points) * 100 if total_points > 0 else 0
    
    # Plot the results
    scatter = ax.scatter(valid_a, valid_b, c=valid_colors, s=10)
    
    # Add coordinate axes
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Draw the boundary circle
    boundary_circle = Circle((0, 0), 100, fill=False, color='black')
    ax.add_patch(boundary_circle)
    
    # Add labels
    ax.set_title(f'sRGB Gamut in CIE LAB Space (L*={L}), Coverage: {coverage:.1f}%')
    ax.set_xlabel('a* (green to red)')
    ax.set_ylabel('b* (blue to yellow)')
    
    # Add color direction labels
    ax.text(0, 95, "YELLOW", ha='center', fontsize=10)
    ax.text(95, 0, "RED", ha='center', fontsize=10)
    ax.text(0, -95, "BLUE", ha='center', fontsize=10)
    ax.text(-95, 0, "GREEN", ha='center', fontsize=10)
    
    plt.tight_layout()
    
    return coverage, fig

# Example usage
if __name__ == "__main__":
    # Create a LAB color wheel with L*=75
    fig1 = create_lab_color_wheel(L=75, resolution=100)
    plt.figure(fig1.number)
    # plt.savefig('lab_color_wheel_L75.png', dpi=300)
    
    # Visualize multiple L* slices
    fig2 = visualize_lab_slices(L_values=[25, 50, 75, 90])
    plt.figure(fig2.number)
    # plt.savefig('lab_multiple_slices.png', dpi=300)
    
    # Calculate gamut coverage at L*=75
    coverage, fig3 = calculate_lab_gamut_coverage(L=75)
    plt.figure(fig3.number)
    # plt.savefig('lab_gamut_coverage_L75.png', dpi=300)
    
    print(f"sRGB gamut coverage at L*=75: {coverage:.1f}%")
    
    plt.show()

# To use the mapping function directly:
# L, a, b = disk_to_lab(0.8, np.pi/4, L=75)  # Maps a point at r=0.8, θ=π/4 to LAB space
