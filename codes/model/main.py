import sys, os, random, importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dgl
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.config import get_config
from model.CURE import CURE
from utils.make_train_datas_1w import make_train_datas
from utils.make_test_datas_1w import make_test_datas
from utils.time_split_batch import split_train_graph
from utils.ns_indexing import ns_indexing
from utils.EarlyStopping import EarlyStopping
from utils.evaluate import ndcg_score, mrr_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


os.environ["WANDB_API_KEY"] = ""

cfg = get_config()

random_seed = cfg.seed
random.seed(random_seed)
np.random.seed(random_seed); torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    torch.cuda.set_device(cfg.gpu_num)
    device = torch.device(f"cuda:{cfg.gpu_num}" if torch.cuda.is_available() else "cpu")
    original_batch_size = cfg.batch_size
    
    # calculate number of snapshots
    interval_minutes = cfg.interval_minutes
    interval_hours = interval_minutes / 60
    snapshot_weeks = 6/7   # history + train
    snapshots_num = int(snapshot_weeks * 7 * 24 / interval_hours)   
    
    # history + train snapshots
    g, splitted_g = split_train_graph(
        snapshot_weeks,
        interval_hours, 
        f'Adressa_1w/datas/total_graph_full_reciprocal_{interval_minutes}m.bin'
    )

    datas = make_train_datas(interval_minutes=interval_minutes)
    train_news, train_category, train_time = zip(*datas)

    word2int = pd.read_csv(os.path.join('data', 'word2int.tsv'), sep='\t')
    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('data/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)   # (330900, 100)
    
    def tokenize_title(title: str) -> list:
        return title.split()

    file_path = 'Adressa_1w/datas/1w_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
    df['click_time'] = pd.to_datetime(df['click_time'])
    df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    combined_news_df = pd.read_csv(
        'Adressa_1w/datas/all_news.tsv',   
        sep='\t'
    ).rename(columns={'newsId': 'clicked_news'})
    
    all_news_ids = pd.read_csv('Adressa_1w/datas/news2int.tsv', sep='\t')['news_id']
    news_num = len(all_news_ids)
    user2int_df = pd.read_csv(os.path.join('Adressa_1w/datas/', 'user2int.tsv'), sep='\t')
    user_num = len(user2int_df['user_int'])
    all_users = [i for i in range(user_num)]
    
    # aggregate news informations per news
    news_info = combined_news_df.groupby('clicked_news', as_index=False).agg({
        'title': 'first',
        'category': 'first',
        'subcategory': 'first'
    })

    # title -> token -> index
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [word_to_idx[w.lower()] if w.lower() in word_to_idx else 0 for w in words]
    )
    
    # category, subcategory -> index
    category2int = pd.read_csv('Adressa_1w/datas/category2int.tsv', sep='\t')   
    cat_num = cfg.num_categories
    
    category_map = category2int.set_index('category')['int'].to_dict()
    news_info['category_idx'] = news_info['category'].map(category_map)
    news_info['subcategory_idx'] = news_info['subcategory'].map(category_map)

    news_info_df = news_info[['clicked_news', 'title_idx', 'category_idx', 'subcategory_idx']].rename(
        columns={'clicked_news': 'news_id'}   
    )

    news_id_to_info = news_info_df.set_index('news_id')\
        [['title_idx', 'category_idx', 'subcategory_idx']].to_dict(orient='index')

    ### Loading idx_infos for calculating NLL loss
    train_ns_idx_batch, _, train_user_idx_batch = ns_indexing('Adressa_1w/train/train_ns.tsv', original_batch_size, user_num=user_num)

    test_datas = make_test_datas(snapshots_num=snapshots_num)
    test_news, test_time, test_empty_check = zip(*test_datas)
    test_ns_idx_batch, test_cand_score_weight_batch, test_user_idx_batch = ns_indexing('Adressa_1w/test/test_ns.tsv', original_batch_size, user_num=user_num, test=True)
    
    print("data loading finished!")
    
    learning_rate = 0.0001
    num_epochs = 10
    batch_size = original_batch_size
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    emb_dim = cfg.word_embedding_dim*3   # 300
    history_length = 100

    # wandb 
    wandb.init(
        project="", 
        config={
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": original_batch_size,
        "emb_dim": emb_dim,
        "history_length": history_length,
        "snapshot_weeks": snapshot_weeks,
        "snapshots_num": snapshots_num,
        }, 
        name="")
    wandb.watch(model, log="all")
    
    # Initialize model
    model = CURE(
        all_news_ids,
        news_id_to_info,
        user_num,
        cat_num,
        news_num,
        pretrained_word_embedding=pretrained_word_embedding,
        emb_dim=emb_dim,
        batch_size=batch_size,
        snapshots_num=snapshots_num,
        config=cfg
    ).to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.01)   
    
    all_losses = []

    best_auc = 0.0
    best_mrr = 0.0
    best_ndcg5 = 0.0
    best_ndcg10 = 0.0

    early_stopper = EarlyStopping(
        emb_dim=emb_dim,      
        patience=3,           
        min_delta=1e-4,
        ckpt_dir=f'Adressa_{cfg.week}w/1_hop_ckpt/bs_{original_batch_size}_lr_{learning_rate}_seed_{random_seed}', 
        verbose=True,
        save_all=False        
    )

    print(f"# of batch: {batch_num}, # of user: {user_num}, "
          f"batch size: {batch_size}, lr: {learning_rate}, "
          f"embedding dim: {emb_dim}, history_length: {history_length}",
          f"window size: {round(cfg.interval_minutes/60, 2)}h",
          f"snapshots_num: {snapshots_num}",
          f"# of hops: {cfg.hop}\n")
    
    # train
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        prev_batch_cnt = 0
        batch_cnt = 0
        batch_size = original_batch_size
        for b in tqdm(range(batch_num), desc=f'training {epoch} epoch batches'):
            prev_batch_cnt = batch_cnt
            batch_cnt += batch_size

            if batch_cnt > len(train_news):
                batch_cnt = len(train_news)
            
            loss = model(
                all_users[prev_batch_cnt:batch_cnt],
                train_news[prev_batch_cnt:batch_cnt],
                train_category[prev_batch_cnt:batch_cnt],
                train_time[prev_batch_cnt:batch_cnt],
                g,
                splitted_g,
                train_ns_idx_batch[b],
                history_length
            )
            
            loss.backward()   
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()   # update parameter via calculated gradient
            optimizer.zero_grad()   # initialize gradient
            
            all_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_samples += 1
            
        epoch_loss = epoch_loss_sum / (epoch_samples if epoch_samples else 1)
        print(f"[Epoch {epoch}] avg train_loss={epoch_loss:.6f}")
            
        # test
        model.eval()
        with torch.no_grad():
            test_batch_num = user_num // original_batch_size \
                             if user_num % original_batch_size == 0 \
                             else user_num // original_batch_size + 1

            all_scores = []
            all_labels = []
            list_mrr = []
            list_ndcg5 = []
            list_ndcg10 = []
            prev_test_batch_cnt = 0
            test_batch_cnt = 0
            empty_batch_count = 0

            for test_b in tqdm(range(test_batch_num), desc=f'Testing Epoch {epoch}', miniters=5, leave=False):
                prev_test_batch_cnt = test_batch_cnt
                test_batch_cnt += original_batch_size
                if test_batch_cnt > len(test_news):
                    test_batch_cnt = len(test_news)

                if not any(test_empty_check[prev_test_batch_cnt:test_batch_cnt]):
                    empty_batch_count += 1
                    continue

                candidate_score, test_loss = model.inference(
                    all_users[prev_test_batch_cnt:test_batch_cnt],
                    test_news[prev_test_batch_cnt:test_batch_cnt],
                    test_time[prev_test_batch_cnt:test_batch_cnt],
                    g,
                    splitted_g,
                    test_ns_idx_batch[test_b],
                    history_length
                )

                candidate_score = candidate_score.cpu().numpy()
                if cfg.adjust_score:
                    test_cand_score_weight = np.array(test_cand_score_weight_batch[test_b])
                    assert candidate_score.shape == test_cand_score_weight.shape
                for i in range(candidate_score.shape[0]):
                    y_score = candidate_score[i]
                    if cfg.adjust_score:
                        y_score = y_score*test_cand_score_weight[i]

                    y_true = np.zeros(len(y_score), dtype=int)
                    y_true[0] = 1

                    all_scores.extend(y_score)
                    all_labels.extend(y_true)

                    list_mrr.append(mrr_score(y_true, y_score))
                    list_ndcg5.append(ndcg_score(y_true, y_score, k=5))
                    list_ndcg10.append(ndcg_score(y_true, y_score, k=10))

            # Test Metrics
            if len(set(all_labels)) > 1:
                final_auc = roc_auc_score(all_labels, all_scores)
            else:
                final_auc = 0.0  

            final_mrr = np.mean(list_mrr) if list_mrr else 0.0
            final_ndcg5 = np.mean(list_ndcg5) if list_ndcg5 else 0.0
            final_ndcg10 = np.mean(list_ndcg10) if list_ndcg10 else 0.0

            avg_metric = (final_auc + final_mrr + final_ndcg5 + final_ndcg10) / 4.0
            print(f"\n[Epoch {epoch} Test Metrics]")
            print(f"AUC={final_auc:.4f}, MRR={final_mrr:.4f}, "
                  f"nDCG@5={final_ndcg5:.4f}, nDCG@10={final_ndcg10:.4f}, "
                  f"avg={avg_metric:.4f}, (empty batch={empty_batch_count})\n")

            wandb.log({
                "train_loss": epoch_loss,
                "auc": final_auc,
                "mrr": final_mrr,
                "ndcg5": final_ndcg5,
                "ndcg10": final_ndcg10,
                "avg_score": avg_metric,
                "empty_batch_count": empty_batch_count,
            })
            
            old_best_score = early_stopper.best_score  
            early_stopper(val_score=avg_metric, model=model, epoch=epoch, lr=learning_rate)

            if early_stopper.best_score != old_best_score:
                best_auc = final_auc
                best_mrr = final_mrr
                best_ndcg5 = final_ndcg5
                best_ndcg10 = final_ndcg10

            if early_stopper.early_stop:
                print("[EarlyStopping] Training is stopped.")
                break  

        if early_stopper.early_stop:
            break  

    print("\n=== Training finished. Loading best checkpoint for final report ===")
    if early_stopper.best_ckpt_path is not None and os.path.exists(early_stopper.best_ckpt_path):
        model.load_state_dict(torch.load(early_stopper.best_ckpt_path))
        print(f"[Info] Best checkpoint (epoch={early_stopper.best_epoch}, avg_score={early_stopper.best_score:.4f}) loaded.")
    else:
        print("[Warning] Best checkpoint file not found. Using last model state.")

    print(f"\n[Training Completed] Best Test Performance (epoch={early_stopper.best_epoch}):")
    print(f" - AUC     : {best_auc:.4f}")
    print(f" - MRR     : {best_mrr:.4f}")
    print(f" - nDCG@5  : {best_ndcg5:.4f}")
    print(f" - nDCG@10 : {best_ndcg10:.4f}")
    print(f" - avg     : {early_stopper.best_score:.4f}\n")

    wandb.finish()

if __name__ == "__main__":
    main()
