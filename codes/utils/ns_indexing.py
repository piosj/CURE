import torch
import pandas as pd
from model.config import Config


def ns_indexing(ns_file_path, batch_size, user_num=84989, test=False):
    ns_df = pd.read_csv(ns_file_path, sep='\t')

    all_user_ids = [i for i in range(user_num)]

    prev_batch = 0
    batch = 0
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1

    # Saving negative sample indicies by batch
    ns_idx_batch = []
    test_cand_score_weight_batch = []
    test_cand_remaining_lifetime_batch = []   # for figure 5
    for b in range(batch_num):

        prev_batch = b * batch_size
        batch = min((b+1) * batch_size, user_num)
        batch_user_ids = all_user_ids[prev_batch:batch]   # ex) 0 ~ 499, 500 ~ 999, ..., 84500 ~ 84989
        
        batch_ns_df = ns_df[ns_df['user_int'].isin(batch_user_ids)]

        ns_idx_list = []
        test_cand_weight_list = []
        test_cand_remaining_lifetime_list = []
        for _, row in batch_ns_df.iterrows():
            pos = int(row['news_int'])
            neg_str = row['negative_samples']
            
            neg_list = [int(x) for x in neg_str.split()]
            negs = neg_list
            ns_idx_list.append([pos] + negs)
            
            user = int(row['user_int'])
                
            if test and Config.adjust_score:
                # score weight 
                candidate_weight_str = row['candidate_weight']
                
                candidate_weight_list = [float(x) for x in candidate_weight_str.split()]
                test_cand_weight_list.append(candidate_weight_list)
        
        # shape: [row_num, 5]
        ns_idx_tensor = torch.tensor(ns_idx_list, dtype=torch.long)
        ns_idx_batch.append(ns_idx_tensor)
        
        if test and Config.adjust_score:
            test_cand_score_weight_batch.append(test_cand_weight_list)
        
    if not test:
        return ns_idx_batch, test_cand_score_weight_batch 
    else:
        return ns_idx_batch, test_cand_score_weight_batch#, test_cand_remaining_lifetime_batch
