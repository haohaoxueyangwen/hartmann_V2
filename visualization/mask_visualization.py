from __future__ import annotations
"""
Mask Visualization Module

Generate visual representations of Hartmann and Microlens array masks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from plot_style import apply_plot_font_bump

apply_plot_font_bump()
import config_simulate as config


def create_mask_visualization(output_path: str) -> None:
    """
    Create a visualization of the current mask configuration.
    
    Shows either Hartmann aperture array or Microlens array depending on config.
    """
    mask_type = getattr(config, 'MASK_TYPE', 'hartmann')
    
    if mask_type == 'hartmann':
        _create_hartmann_mask_plot(output_path)
    elif mask_type == 'microlens':
        _create_microlens_mask_plot(output_path)
    else:
        print(f"Unknown mask type: {mask_type}")



def _create_hartmann_mask_plot(output_path: str) -> None:
    """Generate Hartmann aperture array visualization."""
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    pitch = config.HARTMANN_PITCH_M
    hole_diameter = config.HARTMANN_HOLE_DIAMETER_M
    window_size = config.WINDOW_SIZE_M
    
    hole_radius = hole_diameter / 2
    
    # Create visualization grid
    # Use a sufficient resolution to resolve holes, but not too huge to crash memory
    # Given the large window, creating a full raster is expensive/impossible at high res.
    # So we will plot using geometric patches (Circles) which is vector-based and efficient.
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw background (opaque)
    rect = plt.Rectangle((-window_size/2*1e6, -window_size/2*1e6), 
                         window_size*1e6, window_size*1e6, 
                         facecolor='black', alpha=1.0)
    ax.add_patch(rect)
    
    # Generate hole centers
    n_holes = int(np.ceil(window_size / pitch))
    if n_holes % 2 == 0: n_holes += 1
    
    rows = np.arange(-n_holes//2, n_holes//2 + 1)
    
    # We only draw holes that are within the window
    radius_limit = window_size / 2
    
    # Use collection for performance
    from matplotlib.collections import PatchCollection
    patches = []
    
    for i in rows:
        for j in rows:
            cx = i * pitch
            cy = j * pitch
            # Check if within window
            if abs(cx) <= radius_limit and abs(cy) <= radius_limit:
                circle = plt.Circle((cx*1e6, cy*1e6), hole_radius*1e6)
                patches.append(circle)
    
    p = PatchCollection(patches, facecolor='white', edgecolor='none')
    ax.add_collection(p)
    
    ax.set_xlim(-window_size/2*1e6, window_size/2*1e6)
    ax.set_ylim(-window_size/2*1e6, window_size/2*1e6)
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Hartmann Mask (pitch={pitch*1e6:.1f}µm, d={hole_diameter*1e6:.1f}µm)', fontsize=14)
    
    # --- Zoom Insets ---
    
    # 1. Center Zoom (showing multiple holes)
    zoom_width = pitch * 4 * 1e6 # Show ~4x4 holes
    
    axins1 = zoomed_inset_axes(ax, 4, loc='lower right')
    
    # Re-draw patches in inset (efficiently)
    # Background
    rect_ins = plt.Rectangle((-zoom_width/2, -zoom_width/2), zoom_width, zoom_width, facecolor='black')
    axins1.add_patch(rect_ins)
    
    # Filter patches for inset to keep it light? No, PatchCollection is fast enough usually.
    # But for inset we might need to recreate collection or add the same one? 
    # Adding same collection to multiple axes is not allowed in mpl. Need new collection.
    
    patches_ins = []
    # Optimization: only add patches near center
    limit_ins = zoom_width * 1.5 * 1e-6 # slightly larger than view
    for i in range(-5, 6): # -5 to 5 grid is enough for center zoom
        for j in range(-5, 6):
             cx = i * pitch
             cy = j * pitch
             circle = plt.Circle((cx*1e6, cy*1e6), hole_radius*1e6)
             patches_ins.append(circle)
             
    p_ins = PatchCollection(patches_ins, facecolor='white', edgecolor='none')
    axins1.add_collection(p_ins)
    
    axins1.set_xlim(-zoom_width/2, zoom_width/2)
    axins1.set_ylim(-zoom_width/2, zoom_width/2)
    axins1.set_xticks([])
    axins1.set_yticks([])
    
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="white", alpha=0.5)
    
    # 2. Corner Zoom (Top Left)
    # Determine top-left coordinate
    tl_x = -window_size/2 + window_size/8
    tl_y = window_size/2 - window_size/8
    
    axins2 = zoomed_inset_axes(ax, 4, loc='upper left', borderpad=2)
    
    # Background
    rect_ins2 = plt.Rectangle((tl_x*1e6 - zoom_width/2, tl_y*1e6 - zoom_width/2), zoom_width, zoom_width, facecolor='black')
    axins2.add_patch(rect_ins2)
    
    patches_ins2 = []
    # Find closest lattice index
    idx_x0 = int(round(tl_x / pitch))
    idx_y0 = int(round(tl_y / pitch))
    
    for i in range(idx_x0 - 5, idx_x0 + 6): 
        for j in range(idx_y0 - 5, idx_y0 + 6):
             cx = i * pitch
             cy = j * pitch
             circle = plt.Circle((cx*1e6, cy*1e6), hole_radius*1e6)
             patches_ins2.append(circle)
             
    p_ins2 = PatchCollection(patches_ins2, facecolor='white', edgecolor='none')
    axins2.add_collection(p_ins2)
    
    axins2.set_xlim(tl_x*1e6 - zoom_width/2, tl_x*1e6 + zoom_width/2)
    axins2.set_ylim(tl_y*1e6 - zoom_width/2, tl_y*1e6 + zoom_width/2)
    axins2.set_xticks([])
    axins2.set_yticks([])
    
    mark_inset(ax, axins2, loc1=3, loc2=1, fc="none", ec="white", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mask visualization: {output_path}")


def _create_microlens_mask_plot(output_path: str) -> None:
    """Generate Microlens array visualization (Phase)."""
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    pitch = config.MICROLENS_PITCH_M
    focal_length = config.MICROLENS_FOCAL_LENGTH_M
    fill_factor = getattr(config, 'MICROLENS_FILL_FACTOR', 1.0)
    lens_shape = getattr(config, 'MICROLENS_SHAPE', 'circle')
    window_size = config.WINDOW_SIZE_M
    
    lenslet_size = pitch * fill_factor
    lenslet_half = lenslet_size / 2
    
    # For full window phase map, rasterization is inevitable but heavy.
    # We will compute a grid with moderate resolution and display it.
    # If window is large, we might need to downsample or tile.
    
    # 2048x2048 is reasonable for plotting
    resolution = 2048 
    x = np.linspace(-window_size/2, window_size/2, resolution)
    y = np.linspace(-window_size/2, window_size/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized phase computation
    # Map to local coordinates
    x_local = np.mod(X + pitch/2, pitch) - pitch/2
    y_local = np.mod(Y + pitch/2, pitch) - pitch/2
    
    r_sq = x_local**2 + y_local**2
    
    if lens_shape.lower() == 'circle':
        mask_region = np.sqrt(r_sq) <= lenslet_half
    else:
        mask_region = (np.abs(x_local) <= lenslet_half) & (np.abs(y_local) <= lenslet_half)
    
    wavelength = 1239.8e-9 / config.PHOTON_ENERGY_EV
    k = 2 * np.pi / wavelength
    
    # Phase profile
    phase = np.zeros_like(X)
    phase[mask_region] = -k * r_sq[mask_region] / (2 * focal_length)
    
    # Wrap
    phase_wrapped = np.angle(np.exp(1j * phase))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(phase_wrapped, extent=[-window_size/2*1e6, window_size/2*1e6,
                                          -window_size/2*1e6, window_size/2*1e6],
                   cmap='twilight', origin='lower')
                   
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Microlens Array Phase (pitch={pitch*1e6:.0f}µm, f={focal_length*1e3:.1f}mm)', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Phase (rad)')
    
    # --- Zoom Insets ---
    
    # 1. Center Zoom
    # Extract sub-region from X, Y array is tricky because imshow uses pixels of the generated image.
    # We can just set axes limits on shared data? No, resolution might be poor if we zoom in too much on a 2k image.
    # But for 4x zoom it should be fine.
    
    zoom_width = pitch * 3 * 1e6
    
    axins1 = zoomed_inset_axes(ax, 4, loc='lower right')
    axins1.imshow(phase_wrapped, extent=[-window_size/2*1e6, window_size/2*1e6,
                                         -window_size/2*1e6, window_size/2*1e6],
                  cmap='twilight', origin='lower')
                  
    axins1.set_xlim(-zoom_width/2, zoom_width/2)
    axins1.set_ylim(-zoom_width/2, zoom_width/2)
    axins1.set_xticks([])
    axins1.set_yticks([])
    
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="white", alpha=0.5)
    
    # 2. Corner Zoom
    tl_x_um = (-window_size/2 + window_size/8) * 1e6
    tl_y_um = (window_size/2 - window_size/8) * 1e6
    
    axins2 = zoomed_inset_axes(ax, 4, loc='upper left', borderpad=2)
    axins2.imshow(phase_wrapped, extent=[-window_size/2*1e6, window_size/2*1e6,
                                         -window_size/2*1e6, window_size/2*1e6],
                  cmap='twilight', origin='lower')
                  
    axins2.set_xlim(tl_x_um - zoom_width/2, tl_x_um + zoom_width/2)
    axins2.set_ylim(tl_y_um - zoom_width/2, tl_y_um + zoom_width/2)
    axins2.set_xticks([])
    axins2.set_yticks([])
    
    mark_inset(ax, axins2, loc1=3, loc2=1, fc="none", ec="white", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mask visualization: {output_path}")
