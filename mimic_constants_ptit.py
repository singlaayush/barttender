import pandas as pd
from pathlib import Path
import os
import re
import glob 
import numpy as np
import pandas as pd
from pathlib import Path

num_classes = 1

# every path returned from this file MUST be a pathlib.Path object
home_out_dir = Path("/home/ays124/mimic/cardiomegaly/")
scratch_dir  = Path('/n/scratch/users/a/ays124/mimic-cxr-jpg/')
nb_group_dir = Path('/n/no_backup2/patel/ays124/mimic-dataset/')

# Import cleaned master dataframe
feature_folder = Path('/home/ays124/mimic/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features/')

def get_master_df(idp=False):
    if idp:
        # (2662, 104)
        return pd.read_pickle('/home/ays124/mimic/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    return pd.read_pickle(feature_folder / 'MIMIC_features_v3.pkl')

def get_cardiomegaly_df(idp=False):
    df = None
    if idp:
        df = pd.read_pickle('/home/ays124/mimic/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    else:
        df = pd.read_pickle(feature_folder / 'MIMIC_features_v3.pkl')
    return df[df['Cardiomegaly'].isin([0, 1])]

# includes those with a high number of NaNs
significant_variables_all = ['age_val', 'RR_mean', 'Chloride_mean', 'Urea_Nitrogren_mean', 'SaO2_mean', \
    'PTT_mean', 'Magnesium_mean', 'PO2_mean', 'PCO2_mean', 'Lactate_mean', 'Phosphate_mean', 'Glucose_mean', 'FiO2_mean', 'PH_mean']

# includes only those with a small number of NaNs
significant_variables = ['age_val', 'Chloride_mean', 'RR_mean', 'Urea_Nitrogren_mean', 'Magnesium_mean', 'Glucose_mean', 'Phosphate_mean', 'Hematocrit_mean']

significant_variables_areas = {
    'image': slice(None, 185),
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

z_scores = {
    "Cardiomegaly": {
        'age': 15.3406,
        'urea': -11.3091,
        'chloride': 11.1641,
        'rr': 6.8814,
        'magnesium': 5.1887,
        'glucose': -3.7193,
        'phosphate': 3.3469,
        'hematocrit': -3.2741
    }
}

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

stats_header_debug = ['filename', 'area', 'mean', 'min', '25th_percentile', 'median', 
               '75th_percentile', 'max', 'std_mean', 'std_median', 'mean_w_frac', 'mean_w_iou', 'fraction_attn_area', 'iou']

k_fold_run_id = {'xray' : 'wqzq428c',
'noise': 'jfpf2s6m',
'blank': 'yxa5wmrg'}

k_fold_test_pred_csv_path = {
    'xray' : Path('/home/ays124/mimic/cardiomegaly/cross-val/densenet-xray-age_chloride_rr_urea_nitrogren_magnesium_glucose_phosphate_hematocrit-idp/fold4-wqzq428c/predictions.test.csv'),
    'noise': Path('/home/ays124/mimic/cardiomegaly/cross-val/densenet-noise-age_chloride_rr_urea_nitrogren_magnesium_glucose_phosphate_hematocrit-idp/fold4-jfpf2s6m/predictions.test.csv'),
    'blank': Path('/home/ays124/mimic/cardiomegaly/cross-val/densenet-blank-age_chloride_rr_urea_nitrogren_magnesium_glucose_phosphate_hematocrit-idp/fold9-yxa5wmrg/predictions.test.csv')
}

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_barcode_order_info(order=None, no_bars=False, nan=False, bar_variables=significant_variables):
    if no_bars:
        return None, 'no_bars'
    if nan:
        bar_variables=significant_variables_all
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

def sanity_check_stats_csv(stats_df, areas=significant_variables_areas):
    total_len = stats_df.shape[0]
    for key in areas.keys():
        len_non_zero = stats_df[(stats_df['area'] == key) & (stats_df['mean'] > 0)].shape[0]
        print(f"For {key}: {len_non_zero} samples out of {total_len} are non-zero.")

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
    df['path_preproc'] = df['path'].apply(construct_preproc_path_str, args=(data_dir, preproc_dir))
    return df

def get_merged_stats_sample_df(image_type, suffix, mask_type, label, split, idp):
    df = get_master_df(idp)
    stats_df = get_mask_stats_csv(image_type, suffix, mask_type, label)
    preproc_dir = get_preproc_subpath(image_type, suffix)
    root_dir = get_correct_root_dir(preproc_dir)
    df = df_add_preproc_path_col(df, root_dir, preproc_dir)
    df['path_mask'] = df['path_preproc'].apply(lambda path_preproc: (get_mask_save_dir_path(image_type, suffix, mask_type, label) / os.path.basename(path_preproc).replace('.jpg', '.npy')).as_posix())
    merged_df = stats_df.merge(df.rename(columns={'path_mask': 'filename'}), how='left', on='filename')
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

def standardize_mimic_ethnicity(df):
    # Mapping of original ethnicities to standardized categories
    ethnicity_mapping = {
        "WHITE": "White",
        "WHITE - OTHER EUROPEAN": "White",
        "WHITE - RUSSIAN": "White",
        "WHITE - EASTERN EUROPEAN": "White",
        "WHITE - BRAZILIAN": "White",
        "BLACK/AFRICAN AMERICAN": "Black",
        "BLACK/CAPE VERDEAN": "Black",
        "BLACK/CARIBBEAN ISLAND": "Black",
        "BLACK/AFRICAN": "Black",
        "ASIAN": "Asian",
        "ASIAN - CHINESE": "Asian",
        "ASIAN - SOUTH EAST ASIAN": "Asian",
        "ASIAN - ASIAN INDIAN": "Asian",
        "ASIAN - KOREAN": "Asian",
        "HISPANIC/LATINO - PUERTO RICAN": "Hispanic/Latino",
        "HISPANIC/LATINO - DOMINICAN": "Hispanic/Latino",
        "HISPANIC/LATINO - GUATEMALAN": "Hispanic/Latino",
        "HISPANIC/LATINO - SALVADORAN": "Hispanic/Latino",
        "HISPANIC OR LATINO": "Hispanic/Latino",
        "HISPANIC/LATINO - MEXICAN": "Hispanic/Latino",
        "HISPANIC/LATINO - HONDURAN": "Hispanic/Latino",
        "HISPANIC/LATINO - CUBAN": "Hispanic/Latino",
        "HISPANIC/LATINO - COLUMBIAN": "Hispanic/Latino",
        "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic/Latino",
        "SOUTH AMERICAN": "Hispanic/Latino",
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Asian",
        "AMERICAN INDIAN/ALASKA NATIVE": "Other",
        "MULTIPLE RACE/ETHNICITY": "Other",
        "OTHER": "Other",
        "UNKNOWN": "Other",
        "UNABLE TO OBTAIN": "Other",
        "PATIENT DECLINED TO ANSWER": "Other",
        "PORTUGUESE": "Other"
    }

    df['ethnicity'] = df['ethnicity'].map(ethnicity_mapping)

    return df