from __future__ import annotations
import numpy as np

def create_hartmann_mask(X: np.ndarray, Y: np.ndarray, hole_pitch: float, hole_diameter: float, rotation_deg: float = 0.0):
    """
    Creates a binary Hartmann mask (grid of holes).
    
    Args:
        X, Y: Grid coordinates.
        hole_pitch: Distance between hole centers.
        hole_diameter: Diameter of each hole.
        rotation_deg: Rotation of the grid in degrees.
    """
    # Rotate coordinates to simulate mask rotation
    theta = np.radians(rotation_deg)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Determine hole centers relative to the rotated grid
    # We use modulo to find distance to nearest hole center in the infinite grid
    # Shift by half pitch to center holes
    dist_x = np.mod(X_rot + hole_pitch / 2, hole_pitch) - (hole_pitch / 2)
    dist_y = np.mod(Y_rot + hole_pitch / 2, hole_pitch) - (hole_pitch / 2)
    
    radius = hole_diameter / 2
    mask = (dist_x**2 + dist_y**2) <= radius**2
    
    return mask.astype(float)
