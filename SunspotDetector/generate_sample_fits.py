#!/usr/bin/env python3
"""
Generate Sample FITS Files for Solar Image Analysis

This script creates synthetic FITS files that simulate solar images with sunspots
for testing the solar_analysis.py code. These are not real astronomical data but
provide a way to test the sunspot and limb detection algorithms.

Usage:
    python generate_sample_fits.py

Author: Solar Analysis Team
Date: March 2025
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_solar_disk(size=512, center=None, radius=None, noise_level=0.05):
    """Create a synthetic solar disk image."""
    if center is None:
        center = (size // 2, size // 2)
    if radius is None:
        radius = size // 3
    
    # Create coordinate grid
    y, x = np.ogrid[:size, :size]
    
    # Calculate distance from center for each pixel
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create basic solar disk with limb darkening effect
    # Limb darkening approximated with a simple cos(theta) function
    solar_disk = np.zeros((size, size))
    mask = dist_from_center <= radius
    
    # Apply limb darkening effect
    solar_disk[mask] = 1.0 - 0.3 * (dist_from_center[mask] / radius)**2
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (size, size))
        solar_disk += noise
        solar_disk = np.clip(solar_disk, 0, 1)
    
    # Smooth the image slightly
    solar_disk = gaussian_filter(solar_disk, sigma=1.0)
    
    return solar_disk, center, radius

def add_sunspots(image, center, radius, num_spots=10, spot_size_range=(5, 20)):
    """Add synthetic sunspots to the solar disk."""
    size = image.shape[0]
    sunspot_mask = np.zeros_like(image, dtype=bool)
    
    # Create a list to store sunspot properties
    sunspot_props = []
    
    for _ in range(num_spots):
        # Random position within 80% of the radius
        r = np.random.uniform(0, 0.8 * radius)
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Convert to cartesian coordinates
        spot_x = int(center[0] + r * np.cos(theta))
        spot_y = int(center[1] + r * np.sin(theta))
        
        # Random spot size
        spot_size = np.random.uniform(*spot_size_range)
        
        # Random intensity reduction (darker spots)
        intensity_reduction = np.random.uniform(0.3, 0.8)
        
        # Create spot mask
        y, x = np.ogrid[:size, :size]
        spot_dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
        spot_mask = spot_dist <= spot_size
        
        # Apply spot to image with a soft edge
        # Create a gradient for the spot edge
        spot_gradient = 1.0 - np.exp(-(spot_dist - spot_size)**2 / (2 * (spot_size/3)**2))
        spot_gradient = np.clip(spot_gradient, 0, 1)
        
        # Only modify pixels within the solar disk
        disk_mask = np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius
        valid_mask = spot_mask & disk_mask
        
        # Apply the spot with its gradient
        image[valid_mask] *= (1 - intensity_reduction * (1 - spot_gradient[valid_mask]))
        
        # Update sunspot mask
        sunspot_mask[valid_mask] = True
        
        # Store sunspot properties
        sunspot_props.append({
            'x': spot_x,
            'y': spot_y,
            'size': spot_size,
            'intensity_reduction': intensity_reduction
        })
    
    return image, sunspot_mask, sunspot_props

def save_fits_file(image, filename, header_info=None):
    """Save the image as a FITS file with optional header information."""
    # Create a primary HDU
    hdu = fits.PrimaryHDU(image)
    
    # Add header information
    if header_info:
        for key, value in header_info.items():
            hdu.header[key] = value
    
    # Add basic header info
    hdu.header['DATE-OBS'] = '2025-03-26T12:00:00'
    hdu.header['TELESCOP'] = 'SynthSolar Telescope'
    hdu.header['INSTRUME'] = 'Synthetic Imager'
    hdu.header['OBSERVER'] = 'SolarAnalysisTest'
    hdu.header['COMMENT'] = 'This is a synthetic solar image for testing'
    
    # Create a HDUList and write to file
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    print(f"Saved FITS file: {filename}")

def visualize_sample(image, sunspot_mask, center, radius, filename=None):
    """Visualize the synthetic solar image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Synthetic Solar Image')
    circle = plt.Circle(center, radius, fill=False, color='red')
    axes[0].add_patch(circle)
    
    # Sunspot mask
    axes[1].imshow(image, cmap='gray')
    # Create an overlay of the sunspots
    sunspot_overlay = np.zeros_like(image)
    sunspot_overlay[sunspot_mask] = 1
    axes[1].imshow(sunspot_overlay, cmap='autumn', alpha=0.7)
    axes[1].set_title('Sunspot Locations')
    
    for ax in axes:
        ax.set_axis_off()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
    
    plt.close(fig)

def main():
    """Generate sample FITS files."""
    # Create output directory
    output_dir = 'sample_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample 1: Solar disk with multiple sunspots
    print("Generating sample 1: Solar disk with sunspots...")
    image1, center1, radius1 = create_solar_disk(size=512, noise_level=0.03)
    image1, sunspot_mask1, spots1 = add_sunspots(image1, center1, radius1, num_spots=15)
    
    # Save as FITS
    sample1_file = os.path.join(output_dir, 'sample1_with_sunspots.fits')
    save_fits_file(image1, sample1_file, {
        'SUN_X': center1[0],
        'SUN_Y': center1[1],
        'SUN_R': radius1,
        'NSPOTS': len(spots1)
    })
    
    # Visualize
    vis1_file = os.path.join(output_dir, 'sample1_visualization.png')
    visualize_sample(image1, sunspot_mask1, center1, radius1, vis1_file)
    
    # Sample 2: Solar disk without sunspots
    print("Generating sample 2: Clean solar disk...")
    image2, center2, radius2 = create_solar_disk(size=512, noise_level=0.02)
    
    # Save as FITS
    sample2_file = os.path.join(output_dir, 'sample2_no_sunspots.fits')
    save_fits_file(image2, sample2_file, {
        'SUN_X': center2[0],
        'SUN_Y': center2[1],
        'SUN_R': radius2,
        'NSPOTS': 0
    })
    
    # Visualize
    vis2_file = os.path.join(output_dir, 'sample2_visualization.png')
    visualize_sample(image2, np.zeros_like(image2, dtype=bool), center2, radius2, vis2_file)
    
    # Sample 3: Solar disk with few large sunspots
    print("Generating sample 3: Solar disk with large sunspots...")
    image3, center3, radius3 = create_solar_disk(size=512, noise_level=0.025)
    image3, sunspot_mask3, spots3 = add_sunspots(image3, center3, radius3, 
                                                 num_spots=5, 
                                                 spot_size_range=(15, 30))
    
    # Save as FITS
    sample3_file = os.path.join(output_dir, 'sample3_large_sunspots.fits')
    save_fits_file(image3, sample3_file, {
        'SUN_X': center3[0],
        'SUN_Y': center3[1],
        'SUN_R': radius3,
        'NSPOTS': len(spots3)
    })
    
    # Visualize
    vis3_file = os.path.join(output_dir, 'sample3_visualization.png')
    visualize_sample(image3, sunspot_mask3, center3, radius3, vis3_file)
    
    print("Sample generation complete!")

if __name__ == "__main__":
    main()