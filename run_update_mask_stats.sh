#!/bin/bash

python mimic_update_mask_stats.py --image_type='xray'  --batch_size=64 --mask_type='ig' --idp
python mimic_update_mask_stats.py --image_type='xray'  --batch_size=64 --mask_type='saliency' --idp
python mimic_update_mask_stats.py --image_type='noise' --batch_size=64 --mask_type='ig' --idp
python mimic_update_mask_stats.py --image_type='noise' --batch_size=64 --mask_type='saliency' --idp
python mimic_update_mask_stats.py --image_type='blank' --batch_size=64 --mask_type='ig' --idp
python mimic_update_mask_stats.py --image_type='blank' --batch_size=64 --mask_type='saliency' --idp
