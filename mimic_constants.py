import pandas as pd
from pathlib import Path
import os
import re
import glob 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path

# every path returned from this file MUST be a pathlib.Path object
home_out_dir = Path("/home/ays124/mimic/cardiomegaly/")
scratch_dir  = Path('/n/scratch/users/a/ays124/mimic-cxr-jpg/')
nb_group_dir = Path('/n/no_backup2/patel/ays124/mimic-dataset/')

# Import cleaned master dataframe
feature_folder = Path('/home/ays124/mimic/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features/')

def get_master_df(idp=False):
    if idp:
        # (2375, 104)
        return pd.read_pickle(feature_folder / 'MIMIC_features_with_IDPs.pkl')
    
    return pd.read_pickle(feature_folder / 'MIMIC_features_v3.pkl')

# includes those with a high number of NaNs
significant_variables_all = ['age_val', 'RR_mean', 'Chloride_mean', 'Urea_Nitrogren_mean', 'SaO2_mean', \
    'PTT_mean', 'Magnesium_mean', 'PO2_mean', 'PCO2_mean', 'Lactate_mean', 'Phosphate_mean', 'Glucose_mean', 'FiO2_mean', 'PH_mean']

# includes only those with a small number of NaNs
significant_variables = ['age_val', 'Chloride_mean', 'RR_mean', 'Urea_Nitrogren_mean', 'Magnesium_mean', 'Glucose_mean', 'Phosphate_mean', 'Hematocrit_mean']

# Labels
chart_labels_mean = {
    220045: 'HR_mean',
    220277: 'SpO2_mean',
    223761: 'Temp(F)_mean',
    220210: 'RR_mean',
    220052: 'ABPm_mean',
    220051: 'ABPd_mean',
    220050: 'ABPs_mean',
    220180: 'NBPd_mean',
    220181: 'NBPm_mean',
    220179: 'NBPs_mean',
    223835: 'FiO2_mean',
    220274: 'PH_mean',
    220235: 'PCO2_mean',
    220227: 'SaO2_mean',
    227457: 'PlateletCount_mean',
    227456: 'Albumin_mean',
    220603: 'Cholesterol_mean',
    220645: 'Sodium_mean',
    220224: 'PO2_mean',
}

chart_labels_max = {
    220045: 'HR_max',
    220210: 'RR_max',
    220052: 'ABPm_max',
    220051: 'ABPd_max',
    220050: 'ABPs_max',
    220180: 'NBPd_max',
    220181: 'NBPm_max',
    220179: 'NBPs_max',
    223835: 'FiO2_max',
    220235: 'PCO2_max',
    220645: 'Sodium_max',
}

chart_labels_min = {
    220045: 'HR_min',
    220277: 'SpO2_min',
    220210: 'RR_min',
    220052: 'ABPm_min',
    220051: 'ABPd_min',
    220050: 'ABPs_min',
    220180: 'NBPd_min',
    220181: 'NBPm_min',
    220179: 'NBPs_min',
    220235: 'PCO2_min',
    220645: 'Sodium_min',
}

lab_labels_mean = {
    50826: 'Tidal_Volume_mean',
    51006: 'Urea_Nitrogren_mean',
    50863: 'Alkaline_Phosphatase_mean',
    50893: 'Calcium_Total_mean',
    50902: 'Chloride_mean',
    50931: 'Glucose_mean',
    50813: 'Lactate_mean',
    50960: 'Magnesium_mean',
    50970: 'Phosphate_mean',
    50971: 'Potassium_mean',
    50885: 'Bilirubin',
    51003: 'Troponin-T_mean',
    51221: 'Hematocrit_mean',
    50811: 'Hemoglobin_mean',
    50861: 'ALT_mean',
    50912: 'Creatinine_mean',
    51275: 'PTT_mean',
    51516: 'WBC_mean',
    51214: 'Fibrinogen',
}

lab_labels_max = {
    50971: 'Potassium_max',
    51003: 'Troponin-T_max',
    50811: 'Hemoglobin_max',
    51516: 'WBC_max',
}

lab_labels_min = {
    50971: 'Potassium_min',
    50811: 'Hemoglobin_min',
    51516: 'WBC_min',
}

# Aggregation of all laboratory items into LabItems
LabItems = dict(lab_labels_mean)
LabItems.update(lab_labels_max)
LabItems.update(lab_labels_min)

# Aggregation of the vital signs / chart items into ChartItems
ChartItems = dict(chart_labels_mean)
ChartItems.update(chart_labels_max)
ChartItems.update(chart_labels_min)

indexing_cols = ['subject_id', 'study_id']
imaging_cols  = ['ViewPosition', 'path']
icu_cols = ['hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los', 'Match',\
    'EarlyBoundary', 'PostGapStart', 'PostGapStop']
label_cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', \
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',\
    'Fracture', 'Support Devices']
demographic_cols = ['ethnicity', 'anchor_age', 'anchor_year', 'gender']
chart_labels_mean_cols = list(chart_labels_mean.values())
chart_labels_max_cols  = list(chart_labels_max.values())
chart_labels_min_cols  = list(chart_labels_min.values())
lab_labels_mean_cols   = list(lab_labels_mean.values())
lab_labels_max_cols    = list(lab_labels_max.values())
lab_labels_min_cols    = list(lab_labels_min.values())

stats_header = ['filename', 'area', 'mean', 'min', '25th_percentile', 'median', 
               '75th_percentile', 'max', 'std_mean', 'std_median']

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_barcode_order_info(order=None, bar_variables=significant_variables):
    og_order = np.array(bar_variables)
    order = [int(x) for x in order.strip('[]').split(',')] if isinstance(order, str) else order
    order = list(range(len(bar_variables))) if order is None else order
    suffix = '_'.join(og_order[order]).replace('_mean', '').replace('_val', '').lower()
    return order, suffix

def get_preproc_subpath(image_type, suffix):
    return Path(f'preproc_224x224_{image_type}_{suffix}')

def get_mask_save_dir_path(image_type, suffix, mask_type, label):
    return nb_group_dir / f'{image_type}_{suffix}/{mask_type}/{mask_type}_{label.lower().replace(" ", "_")}_maps/'

def get_mask_stats_csv_path(image_type, suffix, mask_type, label):
    return nb_group_dir / f'{image_type}_{suffix}/{mask_type}/{mask_type}_{label.lower().replace(" ", "_")}_stats.csv'

def get_mask_stats_csv(image_type, suffix, mask_type, label):
    return pd.read_csv(get_mask_stats_csv_path(image_type, suffix, mask_type, label), names=stats_header)

def get_correct_root_dir(sub_path):
    if (scratch_dir / sub_path).exists():
        return scratch_dir
    if (nb_group_dir / sub_path).exists():
        return nb_group_dir
    return ''

def construct_preproc_path(og_path: str, root_dir: Path, preproc_dir=None):
    split = og_path.split("/")
    preproc_filename = f"{split[2]}_{split[3]}_{split[4]}"
    if preproc_dir:
        return root_dir / preproc_dir / preproc_filename
    else:
        return root_dir / preproc_filename

def construct_preproc_path_str(og_path: str, root_dir: Path, preproc_dir=None):
    return construct_preproc_path(og_path, root_dir, preproc_dir).as_posix()

def df_add_preproc_path_col(df, data_dir, preproc_dir):
    df['path_preproc'] = df['Path'].apply(construct_preproc_path_str, args=(data_dir, preproc_dir))
    return df

def get_merged_stats_sample_df(image_type, suffix, mask_type, label, split, idp):
    df = get_master_df(idp)
    stats_df = get_mask_stats_csv(image_type, suffix, mask_type, label)
    preproc_dir = get_preproc_subpath(image_type, suffix)
    root_dir = get_correct_root_dir(preproc_dir)
    df = df_add_preproc_path_col(df, root_dir, preproc_dir)
    merged_df = stats_df.merge(df.rename(columns={'path_preproc': 'filename'}), how='left', on='filename')
    return merged_df

def get_newest_wandb_run_id(img_type_order):
    # Pattern to match directories
    pattern = re.compile(rf'densenet-{re.escape(img_type_order)}-(.+)')
    
    # List to store matching directories with their modification times
    directories = []
    
    # Iterate over all directories in the base path
    for entry in os.scandir(home_out_dir):
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                # Append directory name and modification time
                directories.append((entry.name, entry.stat().st_mtime))
    
    if not directories:
        return None, None
    
    # Sort directories by modification time (newest first)
    directories.sort(key=lambda x: x[1], reverse=True)
    
    # Get the newest directory
    newest_dir = directories[0][0]
    
    # Extract the run id from the directory name
    run_id = pattern.match(newest_dir).group(1)
    
    return run_id

def get_checkpoint_path(image_type: str, suffix: str, run_id: str = None):
    wandb_run_id = run_id if run_id else get_newest_wandb_run_id(f'{image_type}-{suffix}')
    checkpoint_files = list((home_out_dir / f"mimic/{wandb_run_id}/checkpoints/").glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No trained checkpoint found for wandb run id {wandb_run_id}")
    # If there are multiple checkpoints, select the checkpoint with the latest step
    checkpoint_file = sorted(checkpoint_files, key=lambda x: int(x.as_posix().split('step=')[1].split('.ckpt')[0]))[-1]
    return checkpoint_file  # type: Path

class DenseNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.densenet121()
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)
