# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FREEDOM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
        'text_feature_file': 'fake_text_feat.npy' # 修改text文件名 让其定位不到文件
    }
    args, _ = parser.parse_known_args()
    print('===============image modality only========================')
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
