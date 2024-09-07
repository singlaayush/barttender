#!/bin/bash

python mimic_update_mask_stats.py --image_type='xray'  --batch_size=100 --mask_type='ig' --idp --debug
python mimic_update_mask_stats.py --image_type='xray'  --batch_size=100 --mask_type='saliency' --idp --debug
python mimic_update_mask_stats.py --image_type='blank' --batch_size=100 --mask_type='ig' --idp --debug
python mimic_update_mask_stats.py --image_type='blank' --batch_size=100 --mask_type='saliency' --idp --debug
python mimic_update_mask_stats.py --image_type='noise' --batch_size=100 --mask_type='ig' --idp --debug
python mimic_update_mask_stats.py --image_type='noise' --batch_size=100 --mask_type='saliency' --idp --debug
