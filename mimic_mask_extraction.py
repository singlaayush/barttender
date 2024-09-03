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

def pick_attribution(mask_type):
    if mask_type == 'saliency':
        return InputAttribution
    if mask_type == 'ig':
        return IntegratedGradients
    raise ValueError(f"Invalid mask_type '{mask_type}'. Must be 'saliency' or 'ig'.")

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

def process_batch(batch_paths, infl, data_dir, preproc_dir, save_dir, batch_size):
    batch = []
    batch_filenames = []
    
    for og_path in batch_paths:
        # Load and preprocess the image
        path = construct_preproc_path_str(og_path, data_dir, preproc_dir)

        img = imread(path)
        x_pp = torch.from_numpy(img.astype(np.float32))
        x_pp = x_pp.permute(2, 0, 1)
        
        batch.append(x_pp)
        batch_filenames.append(path)
        
        if len(batch) == batch_size:
            yield torch.stack(batch), batch_filenames
            batch.clear()
            batch_filenames.clear()
    
    if batch:
        yield torch.stack(batch), batch_filenames

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

def process_and_save_stats(attrs_input, batch_filenames, save_dir, stats_file, image_type):
    # threshold and blur the saliency maps
    masks = normalize(attrs_input, blur=5, threshold=0.9, masked_opacity=0.0)
    
    # Define areas of interest
    areas = {
        f'{image_type}': slice(None, 185),
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
    for idx, (raw_mask, filename) in enumerate(zip(attrs_input, batch_filenames)):
        # save the raw saliency map
        base_filename = os.path.basename(filename).replace('.jpg', '.npy')
        save_path = os.path.join(save_dir, base_filename)
        np.save(save_path, raw_mask)

        # load the normalized saliency map
        mask = masks[idx]
        
        # Compute stats for each area
        for area_name, area_slice in areas.items():
            area_data = mask[:, area_slice, :].flatten()  # CHW
            area_stats = compute_stats(area_data)
            stats.append([filename, area_name] + area_stats.tolist())
    
    # Convert to DataFrame and append to CSV
    df = pd.DataFrame(stats, columns=stats_header)
    df.to_csv(stats_file, mode='a', header=not os.path.exists(stats_file), index=False)

def main(image_type, df, attribution, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, checkpoint_file):
    total_images = len(df)

    pytorch_model = DenseNet.load_from_checkpoint(
        checkpoint_file, 
        num_classes=1
    )
    model = get_model_wrapper(pytorch_model)
    infl = attribution(model, qoi=ClassQoI(0), resolution = 10)
    
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for batch, batch_filenames in process_batch(df['path'], infl, data_dir, preproc_dir, save_dir, batch_size):
            attrs_input = infl.attributions(batch)
            process_and_save_stats(attrs_input, batch_filenames, save_dir, stats_save_path, image_type)
            pbar.update(len(batch_filenames))

def cli(image_type: str = 'xray', order = None, batch_size: int = 64, mask_type: str = 'saliency', label: str = 'cardiomegaly', split: str = 'test', run_id: str = None, idp: bool = False, nan: bool = False, no_bars: bool = False):
    """
    Runs the mask extraction pipeline.
    :param batch_size: Number of images to process in each batch
    :param mask_type: Type of attribution mask to generate ('saliency' or 'ig')
    :param label_idx: ID of CheXpert label to process against -- used for slurm job arrays
    :param label: CheXpert label to process against
    :param split: CheXpert data split to process (pick one from 'test' or 'val')
    """
    attribution = pick_attribution(mask_type)
    
    order, suffix = get_barcode_order_info(order=order, no_bars=no_bars, nan=nan)
    preproc_dir = get_preproc_subpath(image_type, suffix)
    checkpoint_file = get_checkpoint_path(image_type, suffix, run_id)

    exp_dir = home_out_dir / f"cross-val/densenet-{image_type}-{suffix}-{'idp' if idp else 'no_idp'}/"
    df = pd.read_csv(exp_dir / f'{split}.csv')
    
    data_dir = get_correct_root_dir(preproc_dir)
    save_dir = get_mask_save_dir_path(image_type, suffix, mask_type, label)
    stats_save_path = get_mask_stats_csv_path(image_type, suffix, mask_type, label)
    
    os.makedirs(save_dir, exist_ok=True)
    open(stats_save_path, 'w').close()  # create/clear file

    print(f'Image Type: {image_type}')
    print(f"Barcode Order: {suffix.replace('_', ', ')}")
    print(f'save_dir: {save_dir}')

    main(image_type, df, attribution, label, data_dir, preproc_dir, save_dir, stats_save_path, batch_size, checkpoint_file)

if __name__ == '__main__':
    # python mimic_mask_extraction.py --image_type='xray'  --run_id='wqzq428c' --batch_size=1 --mask_type='ig' --idp
    # python mimic_mask_extraction.py --image_type='noise' --run_id='jfpf2s6m' --batch_size=1 --mask_type='ig' --idp
    # python mimic_mask_extraction.py --image_type='blank' --run_id='yxa5wmrg' --batch_size=1 --mask_type='ig' --idp
    fire.Fire(cli)
