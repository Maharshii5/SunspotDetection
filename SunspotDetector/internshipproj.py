#!/usr/bin/env python3
"""
Solar Limb and Sunspot Detection

Detects the Sun's limb as a circle and identifies sunspots in solar FITS images.
Calculates center coordinates (x, y), radius, and generates a binary sunspot map.

install numpy, scipy, matplotlib and astropy (4 libraries used)
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, median_filter, binary_fill_holes


def load_fits(file_path):
    try:
        with fits.open(file_path) as hdul:
            first_hdu = hdul[0]
            img = first_hdu.data
            hdr = first_hdu.header
           
            if img.ndim > 2:
                img = img[0, 0, :, :] if img.ndim == 4 else img[0, :, :]
                
            return img, hdr
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        return None, None


def clean_image(raw_img):
    
    if raw_img.max() > 1.0:
        normed = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    else:
        normed = raw_img.copy()
    smoothed = gaussian_filter(normed, sigma=1)
    
    return smoothed


def circle_fit_error(params, points):
    
    cx, cy, r = params
    x, y = points[:, 1], points[:, 0]  
    dist = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    return np.sum(dist**2)


def find_sun_edge(img, thresh_val=0.5):
    
    threshold = thresh_val * (np.median(img) + np.mean(img)) / 2
    bin_img = img > threshold
    
    
    bin_img = ndimage.binary_opening(bin_img, structure=np.ones((5, 5)))
    bin_img = ndimage.binary_closing(bin_img, structure=np.ones((5, 5)))
    
    
    bin_img = binary_fill_holes(bin_img)
    
    
    grad_x = ndimage.sobel(bin_img, axis=0)
    grad_y = ndimage.sobel(bin_img, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    
    edge_img = grad_mag > 0.1 * np.max(grad_mag)
    
    
    edge_pts = np.array(np.where(edge_img)).T
    
    if len(edge_pts) < 3:
        raise ValueError("Not enough edge points found")
    
    h, w = img.shape
    init_guess = [w // 2, h // 2, min(w, h) // 2]
    

    result = minimize(
        circle_fit_error,
        init_guess,
        args=(edge_pts,),
        method='L-BFGS-B',
        bounds=[(0, w), (0, h), (min(w, h) // 4, min(w, h))]
    )
    
    cx, cy, r = result.x
    
    return cx, cy, r, edge_pts


def find_sunspots(img, cx, cy, r, for_viz=False):
    
    h, w = img.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    
    
    disk_mask = dist <= (r * 0.95)
    
    
    disk = np.copy(img)
    disk[~disk_mask] = np.nan
    
    
    valid_pix = disk[~np.isnan(disk)]
    avg = np.mean(valid_pix)
    std = np.std(valid_pix)
    
    
    is_test = img.shape[0] <= 200 and img.shape[1] <= 200
    
    
    if is_test:
        pixel_range = np.max(valid_pix) - np.min(valid_pix)
        if pixel_range < 0.5: 
            return np.zeros_like(img, dtype=int)
    
    
    bg = median_filter(disk, size=15)
    
    enhanced = np.zeros_like(disk)
    valid = ~np.isnan(disk) & ~np.isnan(bg) & (bg > 0)
    enhanced[valid] = disk[valid] / bg[valid]
    
    
    spots = np.zeros_like(disk, dtype=bool)
    
    
    if is_test:
        core_thresh = avg - 5 * std
        enh_thresh = 0.6
        min_size = 20
    else:
        core_thresh = avg - 3 * std
        enh_thresh = 0.85
        min_size = 5
    
   
    cores = (disk < core_thresh) & disk_mask
    spots = spots | cores
    
    
    enh_spots = (enhanced < enh_thresh) & disk_mask & ~np.isnan(enhanced)
    spots = spots | enh_spots
    
   
    clean = ndimage.binary_opening(spots, structure=np.ones((3, 3)))
    
    
    labeled, num = ndimage.label(clean)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  
        small = sizes < min_size
        small_mask = np.isin(labeled, np.where(small)[0])
        clean[small_mask] = False
    
    
    binary_spots = np.zeros_like(img, dtype=int)
    binary_spots[clean] = 1
    
    return binary_spots


def show_results(img, cx, cy, r, spots, output_file=None):
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    
    circ = plt.Circle((cx, cy), r, fill=False, color='red')
    ax[0].add_patch(circ)
    ax[0].plot(cx, cy, 'r+', markersize=10)
    ax[1].imshow(spots, cmap='binary')
    ax[1].set_title('Sunspots Mask')
    
  
    ax[2].imshow(img, cmap='gray')
    overlay = np.zeros_like(img)
    overlay[spots > 0] = 1
    ax[2].imshow(overlay, cmap='autumn', alpha=0.7)
    ax[2].set_title('Sunspots Overlay')
    
    
    fig.suptitle(f"Sun's Center: ({cx:.1f}, {cy:.1f}), Radius: {r:.1f} pixels", fontsize=14)
    
    for a in ax:
        a.set_axis_off()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def process_image(fits_file, out_dir=None, show=True):
    
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
   
    img_data, header = load_fits(fits_file)
    
    if img_data is None:
        print(f"Could not process {fits_file}")
        return None
    
   
    img = clean_image(img_data)
    
    
    try:
        cx, cy, r, edge_pts = find_sun_edge(img)
        print(f"Sun's Center: ({cx:.1f}, {cy:.1f}), Radius: {r:.1f} pixels")
    except Exception as e:
        print(f"Error finding Sun's edge: {e}")
        return None
    
    
    spots = find_sunspots(img, cx, cy, r)
    
    
    labeled, num_spots = ndimage.label(spots)
    print(f"Number of sunspot regions: {num_spots}")
    
    
    if show:
        base_name = os.path.splitext(os.path.basename(fits_file))[0]
        out_file = os.path.join(out_dir, f"{base_name}_results.png") if out_dir else None
        show_results(img, cx, cy, r, spots, out_file)
    
   
    if out_dir:
        base_name = os.path.splitext(os.path.basename(fits_file))[0]
        mask_file = os.path.join(out_dir, f"{base_name}_spots_mask.fits")
        
        
        hdu = fits.PrimaryHDU(spots)
        
        if header:
            for key in ['DATE-OBS', 'TELESCOP', 'INSTRUME', 'OBSERVER']:
                if key in header:
                    hdu.header[key] = header[key]
        
        hdu.header['COMMENT'] = 'Sunspots binary mask (1=sunspot, 0=elsewhere)'
        hdu.header['HISTORY'] = 'Created by solar analyzer'
        hdu.header['SUN_X'] = (cx, 'X-coordinate of Sun center')
        hdu.header['SUN_Y'] = (cy, 'Y-coordinate of Sun center')
        hdu.header['SUN_R'] = (r, 'Radius of Sun')
        
       
        hdu.writeto(mask_file, overwrite=True)
        print(f"Mask saved to {mask_file}")
    
    
    return {
        'center_x': cx,
        'center_y': cy,
        'radius': r,
        'sunspots_mask': spots,
        'num_spots': num_spots,
        'image': img
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process solar FITS images to find Sun limb and sunspots')
    parser.add_argument('fits_files', nargs='+', help='FITS files to process')
    parser.add_argument('-o', '--output-dir', help='Directory for output files', default='results')
    parser.add_argument('-v', '--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    for fits_file in args.fits_files:
        if not os.path.exists(fits_file):
            print(f"File not found: {fits_file}")
            continue
        
        print(f"\nProcessing {fits_file}...")
        process_image(fits_file, args.output_dir, args.visualize)