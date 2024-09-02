import pandas as pd
from pathlib import Path

# Import cleaned master dataframe
feature_folder = Path('/home/ays124/mimic/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features/')

def get_master_df(idp=False):
    if idp:
        # (2375, 104)
        return pd.read_pickle(feature_folder / 'MIMIC_features_with_IDPs.pkl')
    
    return pd.read_pickle(feature_folder / 'MIMIC_features_v4.pkl')

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
