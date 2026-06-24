import pandas as pd
import numpy as np
import dgl
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from dgl.data.utils import save_graphs

# save graphs with different time window sizes
interval_minutes = [30, 720, 1440, 2160]

for interval_minute in interval_minutes:
    # (1) the function of calculating interval idx
    def get_period_start(click_time, interval_minutes, start_time=datetime.time(0, 0, 0)):
        base_start = datetime.datetime.combine(click_time.date(), start_time)
        if click_time < base_start:
            base_start -= datetime.timedelta(days=1)
        delta = click_time - base_start
        periods = int(delta.total_seconds() // (interval_minutes * 60))
        return base_start + datetime.timedelta(minutes=interval_minutes * periods)


    # (2) data loading
    history_data_path = 'Adressa_1w/datas/1w_behaviors.tsv'
    df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
    df = df.dropna(subset=['clicked_news'])

    df['click_time'] = pd.to_datetime(df['click_time'])

    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-11 00:00:00')
    df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]

    # (2-1) Calculate time interval index('Period_Start')
    df['Period_Start'] = df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=interval_minute))
    # period_start -> time_idx mapping(start to 0)
    unique_period_starts = df['Period_Start'].unique()
    time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
    df['time_idx'] = df['Period_Start'].map(time_dict)

    # (2-2) Applying news2int 
    news2int = pd.read_csv('Adressa_1w/datas/news2int.tsv', sep='\t')
    df['clicked_news'] = df['clicked_news'].astype(str).str.strip()
    news2int['news_id'] = news2int['news_id'].astype(str).str.strip()
    df = pd.merge(df, news2int, left_on='clicked_news', right_on='news_id', how='left')
    df.drop(columns=['news_id'], inplace=True)

    # (2-3) Applying user2int
    user2int_df = pd.read_csv(os.path.join('Adressa_1w/datas/', 'user2int.tsv'), sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    df['user_int'] = df['history_user'].map(user2int)

    # (2-4) Applying category2int
    category2int = pd.read_csv('Adressa_1w/datas/category2int_nyheter_splitted.tsv', sep='\t')
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)

    cat2int = category2int.set_index('category')['int'].to_dict()
    def get_cat_int(row):
        if row['category'] == 'nyheter':
            return cat2int.get(row['subcategory'], cat2int['No category'])
        else:
            return cat2int.get(row['category'], cat2int['No category'])
    
    # (2-5) To use nyheter's subcategories as categories
    nyheter_mask = df['category'] == 'nyheter'
    df.loc[nyheter_mask, 'category'] = (
        df.loc[nyheter_mask, 'subcategory'] + '_nyheter'
    )
    df['category_int'] = df.apply(get_cat_int, axis=1)
    
    grouped = df.groupby('Period_Start')

    # (2-6) Reset indices of df -> mapping forward edge with rows of df 
    df = df.reset_index(drop=True)

    num_news_nodes = len(news2int) 
    num_user_nodes = len(user2int_df)

    # (3) Generating graph
    src_edges = df['news_int'].values + num_user_nodes   # (forward) news node
    dst_edges = df['user_int'].values   # (forward) user node
    cat_idx = df['category_int'].values    
    edge_time_idx = df['time_idx'].values  


    # forward edges: ('user','clicked','news') = (dst_edges, src_edges)
    # reverse edges: ('news','clicked_reverse','user') = (src_edges, dst_edges)
    g = dgl.DGLGraph()
    g.add_nodes(num_news_nodes + num_user_nodes)
    g.add_edges(src_edges, dst_edges)

    g.edata['cat_idx'] = torch.tensor(cat_idx, dtype=torch.long)
    g.edata['time_idx'] = torch.tensor(edge_time_idx, dtype=torch.long)
    
    # Add reciprocal edges
    g.add_edges(dst_edges, src_edges)
    # Whole edge data: forward + reciprocal
    g.edata['cat_idx'][len(cat_idx):] = torch.tensor(cat_idx, dtype=torch.long)
    g.edata['time_idx'][len(edge_time_idx):] = torch.tensor(edge_time_idx, dtype=torch.long)

    # (3-1) save global graph g
    g_save_path = f"Adressa_1w/datas/total_graph_full_reciprocal_{interval_minute}m.bin"
    save_graphs(g_save_path, [g])

    print(f"Total graph g({g_save_path}) saved!")