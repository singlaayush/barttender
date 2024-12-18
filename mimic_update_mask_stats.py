import os
import re
import csv
import glob
import fire
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from datetime import datetime
from skimage.io import imread
from mimic_constants import *
from scipy.ndimage import gaussian_filter
from trulens.nn.quantities import ClassQoI
from trulens.nn.models import get_model_wrapper
from trulens.nn.attribution import InputAttribution, IntegratedGradients

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="__array_wrap__ must accept context and return_scalar arguments")

def normalize(attributions, blur=None, threshold=None, masked_opacity=0.0) -> np.ndarray:
    """
    Normalize the attributions to be in the range [0, 1], and then threshold them based on %tile
    """
    # Blur the attributions so the explanation is smoother.
    if blur is not None:
        attributions = [gaussian_filter(a, blur) for a in attributions]
    
    # min-max normalization
    attributions = [a - a.min() for a in attributions]
    attributions = [
        0. * a if a.max() == 0. else a / a.max() for a in attributions
    ]
    
    # Threshold the attributions to create a mask.
    masks = None
    if threshold is not None:
        percentiles = [
            np.percentile(a, 100 * threshold) for a in attributions
        ]
        masks = np.array(
            [
                np.where(a > p, a, masked_opacity)
                for a, p in zip(attributions, percentiles)
            ]
        )
    else:
        masks = np.array(attributions)
    return masks

def process_batch(batch_paths, data_dir, preproc_dir, save_dir, batch_size):
    batch = []
    batch_filenames = []
    
    for og_path in batch_paths:
        # Load and preprocess the image
        path = construct_preproc_path_str(og_path, data_dir, preproc_dir)
        path_mask = save_dir / os.path.basename(path).replace('.jpg', '.npy')
        x = np.load(path_mask)
        
        batch.append(x)
        batch_filenames.append(path_mask)
        
        if len(batch) == batch_size:
            yield np.stack(batch), batch_filenames
            batch.clear()
            batch_filenames.clear()
    
    if batch:
        yield np.stack(batch), batch_filenames

def compute_stats(data):
    """Compute statistics for a batch of data."""
    non_zero = data[data != 0]
    if len(non_zero) == 0:
        return np.zeros(8)  # Return zeros if all data is zero
    
    return np.array([
        non_zero.mean(),
        non_zero.min(),
        np.percentile(non_zero, 25),
        np.median(non_zero),
        np.percentile(non_zero, 75),
        non_zero.max(),
        np.std(non_zero),
        np.std(np.abs(non_zero - np.median(non_zero)))
    ])

def compute_weighted_stats(data, data_area, attn_map_area, total_attn_map_area):
    """Compute statistics for a batch of data."""
    non_zero = data[data != 0]
    if attn_map_area == 0:
        return np.zeros(12)  # Return zeros if all data is zero
    meann = non_zero.mean()  # misspelled on purpose to avoid clashes in namespace
    fraction_attn_area = attn_map_area / total_attn_map_area if total_attn_map_area else 0
    iou = attn_map_area / (data_area + total_attn_map_area - attn_map_area)
    return np.array([
        meann,
        non_zero.min(),
        np.percentile(non_zero, 25),
        np.median(non_zero),
        np.percentile(non_zero, 75),
        non_zero.max(),
        np.std(non_zero),
        np.std(np.abs(non_zero - np.median(non_zero))),
        meann * fraction_attn_area,
        meann * iou,
        fraction_attn_area,
        iou
    ])

def process_and_save_stats(attrs_input, batch_filenames, save_dir, stats_file, image_type, suffix, blur, to_ignore, debug):
    # threshold and blur the saliency maps
    masks = normalize(attrs_input, blur=blur, threshold=0.9, masked_opacity=0.0)  # blur default is 5
    
    # Define areas of interest
    areas = get_significant_variables_areas(image_type)
    
    stats = []
    for idx, (mask, filename) in enumerate(zip(masks, batch_filenames)):
        # Compute stats for each area
        total_attn_map_area = np.sum(mask != 0)  # number of non-zero pixels in entire mask
        #print(total_attn_map_area)
        for area_name, area_slice in areas.items():
            area_data = mask[:, area_slice, :].flatten()  # CHW
            attn_map_area = np.sum(area_data != 0) # number of non-zero pixels in region of mask
            area_stats = None
            if debug:
                area_stats = compute_weighted_stats(area_data, len(area_data), attn_map_area, total_attn_map_area)
            else:
                area_stats = compute_stats(area_data)
            stats.append([filename, area_name] + area_stats.tolist())
    header = stats_header_debug if debug else stats_header 
    # Convert to DataFrame and append to CSV
    df = pd.DataFrame(stats, columns=header)
    df.to_csv(stats_file, mode='a', header=not os.path.exists(stats_file), index=False)

def main(image_type, df, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, suffix, blur, to_ignore, debug):
    total_images = len(df)
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for batch, batch_filenames in process_batch(df['path'], data_dir, preproc_dir, save_dir, batch_size):
            process_and_save_stats(batch, batch_filenames, save_dir, stats_save_path, image_type, suffix, blur, to_ignore, debug)
            pbar.update(len(batch_filenames))

def cli(image_type: str = 'xray', order = None, batch_size: int = 64, mask_type: str = 'saliency', label: str = 'cardiomegaly', split: str = 'test', to_ignore: int = 0, blur: int = 5, run_id: str = None, debug: bool = False, idp: bool = False, nan: bool = False, no_bars: bool = False):
    """
    Runs the mask extraction pipeline.
    :param batch_size: Number of images to process in each batch
    :param mask_type: Type of attribution mask to generate ('saliency' or 'ig')
    :param label_idx: ID of CheXpert label to process against -- used for slurm job arrays
    :param label: CheXpert label to process against
    :param split: CheXpert data split to process (pick one from 'test' or 'val')
    """    
    order, suffix = get_barcode_order_info(order=order, no_bars=no_bars, nan=nan)
    preproc_dir = get_preproc_subpath(image_type, suffix)

    exp_dir = home_out_dir / f"cross-val/densenet-{image_type}-{suffix}-{'idp' if idp else 'no_idp'}/"
    df = pd.read_csv(exp_dir / f'{split}.csv')

    data_dir = get_correct_root_dir(preproc_dir)
    save_dir = get_mask_save_dir_path(image_type, suffix, mask_type, label)
    stats_save_path = None
    #if debug:
    #    stats_save_path = f'notebooks/{image_type}.csv'
    #else:

    stats_save_path = get_mask_stats_csv_path(image_type, suffix, mask_type, label)

    open(stats_save_path, 'w').close()  # create/clear file

    print(f'Image Type: {image_type}')
    print(f"Barcode Order: {suffix.replace('_', ', ')}")
    print(f'save_dir: {save_dir}')

    main(image_type, df, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, suffix, blur, to_ignore, debug)

if __name__ == '__main__':
    fire.Fire(cli)
