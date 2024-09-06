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

def process_and_save_stats(attrs_input, batch_filenames, save_dir, stats_file, image_type, suffix, blur, to_ignore):
    # threshold and blur the saliency maps
    masks = normalize(attrs_input, blur=blur, threshold=0.9, masked_opacity=0.0)  # blur default is 5
    
    # Define areas of interest
    areas = {
        f'{image_type}': slice(None, 185 - to_ignore),
	    'age_bar': slice(185, 190),
	    'chloride_bar': slice(190, 194),
	    'rr_bar': slice(194, 198),
	    'urea_bar': slice(198, 202),
	    'nitrogren_bar': slice(202, 207),
	    'magnesium_bar': slice(207, 211),
	    'glucose_bar': slice(211, 215),
	    'phosphate_bar': slice(215, 220),
	    'hematocrit_bar': slice(220, 224),
    }
    
    stats = []
    for idx, (mask, filename) in enumerate(zip(masks, batch_filenames)):
        # Compute stats for each area
        for area_name, area_slice in areas.items():
            area_data = mask[:, area_slice, :].flatten()  # CHW
            area_stats = compute_stats(area_data)
            stats.append([filename, area_name] + area_stats.tolist())
    
    # Convert to DataFrame and append to CSV
    df = pd.DataFrame(stats, columns=stats_header)
    df.to_csv(stats_file, mode='a', header=not os.path.exists(stats_file), index=False)

def main(image_type, df, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, suffix, blur, to_ignore):
    total_images = len(df)
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for batch, batch_filenames in process_batch(df['path'], data_dir, preproc_dir, save_dir, batch_size):
            process_and_save_stats(batch, batch_filenames, save_dir, stats_save_path, image_type, suffix, blur, to_ignore)
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
    if debug:
        stats_save_path = f'notebooks/{image_type}.csv'
    else:
        stats_save_path = get_mask_stats_csv_path(image_type, suffix, mask_type, label)

    open(stats_save_path, 'w').close()  # create/clear file

    print(f'Image Type: {image_type}')
    print(f"Barcode Order: {suffix.replace('_', ', ')}")
    print(f'save_dir: {save_dir}')

    main(image_type, df, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, suffix, blur, to_ignore)

if __name__ == '__main__':
    fire.Fire(cli)
