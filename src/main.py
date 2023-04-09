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
        'gpu_id': 0

    }

    # config_dict = {
    #     "gpu_id": 0,  
    #     "use_gpu": True,  
    #     "seed": [999],  
    #     "data_path": "/root/MMRec/data/",  
    #     "inter_file_name": "baby.inter",
    #     "inter_splitting_label": "x_label",  
    #     "filter_out_cod_start_users": True,  
    #     "is_multimodal_model": True,  
    #     "checkpoint_dir": "saved",  
    #     "save_recommended_topk": True,  
    #     "recommend_topk": "recommend_topk/",  
    #     "embedding_size": 64,  
    #     "weight_decay": 0.0,  
    #     "req_training": True,  
    #     "embedding_size": 3780,  
    #     "epochs": 1,  
    #     "stopping_step": 20,  
    #     "train_batch_size": 2048,  
    #     "learner": "adam",  
    #     "learning_rate": 0.001,  
    #     "learning_rate_scheduler": [1.0, 50],  
    #     "eval_step": 1,  
    #     "training_neg_sample_num": 1,  
    #     "use_neg_sampling": True,  
    #     "use_full_sampling": False,  
    #     "NEG_PREFIX": "neg__",  
    #     # "USER_ID_FIELD": {  
    #     #     "user_id": "token"
    #     # },  
    #     # "ITEM_ID_FIELD": {  
    #     #     "item_id": "token"
    #     # },  
    #     # "TIME_FIELD": {  
    #     #     "timestamp": "float"  
    #     # },  
    #     "USER_ID_FIELD": "userID",  
    #     "ITEM_ID_FIELD": "itemID",  
    #     "TIME_FIELD": "timestamp",  
    #     "field_separator": "\t",  
    #     "metrics": ["Recall", "NDCG", "Precision", "MAP"],  
    #     "topk": [5, 10, 20, 50],  
    #     "valid_metric": "Recall@20",  
    #     "use_raw_features": False,  
    #     "max_txt_len": 32,  
    #     "max_img_size": 256,  
    #     "vocab_size": 30522,  
    #     "type_vocab_size": 2,  
    #     "hidden_size": 4,  
    #     "pad_token_id": 0,  
    #     "max_position_embeddings": 512,  
    #     "layer_norm_eps": 1e-12,  
    #     "hidden_dropout_prob": 0.1,  
    #     "end2end": False,  
    #     "vision_feature_file": 'image_feat.npy',
    #     "text_feature_file": 'text_feat.npy',
    #     "hyper_parameters": ["seed"] ,
    #     'embedding_size': 64, 'feat_embed_dim': 64, 'weight_size': [64, 64],  
    #     'lambda_coeff': 0.9, 'reg_weight': [0.0, 1e-05, 1e-04, 1e-03],  
    #     'n_mm_layers': 1, 'n_ui_layers': 2, 'knn_k': 10,  
    #     'mm_image_weight': 0.1, 'dropout': [0.8, 0.9],  
    #     'hyper_parameters': ['dropout', 'reg_weight'] 

    # }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


