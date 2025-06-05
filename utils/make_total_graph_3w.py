import pandas as pd
import numpy as np
import dgl
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from dgl.data.utils import save_graphs

interval_minutes = [30, 720, 1440, 2160]

for interval_minute in interval_minutes:
    def get_period_start(click_time, interval_minutes, start_time=datetime.time(0, 0, 0)):
        base_start = datetime.datetime.combine(click_time.date(), start_time)
        if click_time < base_start:
            base_start -= datetime.timedelta(days=1)
        delta = click_time - base_start
        periods = int(delta.total_seconds() // (interval_minutes * 60))
        return base_start + datetime.timedelta(minutes=interval_minutes * periods)

    behavior_path = 'behaviors.tsv'
    df = pd.read_csv(behavior_path, sep='\t', encoding='utf-8')
    df = df.dropna(subset=['clicked_news'])

    df['click_time'] = pd.to_datetime(df['click_time'])

    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-23 00:00:00')
    df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]

    df['Period_Start'] = df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=interval_minute))

    unique_period_starts = df['Period_Start'].unique()
    time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
    df['time_idx'] = df['Period_Start'].map(time_dict)

    news2int = pd.read_csv('news2int.tsv', sep='\t')
    df['clicked_news'] = df['clicked_news'].astype(str).str.strip()
    news2int['news_id'] = news2int['news_id'].astype(str).str.strip()
    df = pd.merge(df, news2int, left_on='clicked_news', right_on='news_id', how='left')
    df.drop(columns=['news_id'], inplace=True)

    user2int_df = pd.read_csv('user2int.tsv', sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    df['user_int'] = df['history_user'].map(user2int)

    category2int = pd.read_csv('category2int_nyheter_splitted.tsv', sep='\t')
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)

    cat2int = category2int.set_index('category')['int'].to_dict()
    def get_cat_int(row):
        if row['category'] == 'nyheter':
            return cat2int.get(row['subcategory'], cat2int['No category'])
        else:
            return cat2int.get(row['category'], cat2int['No category'])

    nyheter_mask = df['category'] == 'nyheter'
    df.loc[nyheter_mask, 'category'] = (
        df.loc[nyheter_mask, 'subcategory'] + '_nyheter'
    )
    df['category_int'] = df.apply(get_cat_int, axis=1)
    print(df[['category', 'subcategory', 'category_int']].head(10))

    grouped = df.groupby('Period_Start')
    
    df = df.reset_index(drop=True)

    num_news_nodes = len(news2int) 
    num_user_nodes = len(user2int_df)

    src_edges = df['news_int'].values + num_user_nodes     
    dst_edges = df['user_int'].values      
    cat_idx = df['category_int'].values    
    edge_time_idx = df['time_idx'].values 
    
    g = dgl.DGLGraph()
    g.add_nodes(num_news_nodes + num_user_nodes)
    g.add_edges(src_edges, dst_edges)

    g.edata['cat_idx'] = torch.tensor(cat_idx, dtype=torch.long)
    g.edata['time_idx'] = torch.tensor(edge_time_idx, dtype=torch.long)

    g.add_edges(dst_edges, src_edges)
    g.edata['cat_idx'][len(cat_idx):] = torch.tensor(cat_idx, dtype=torch.long)
    g.edata['time_idx'][len(edge_time_idx):] = torch.tensor(edge_time_idx, dtype=torch.long)

    g_save_path = f"total_graph_full_reciprocal_{interval_minute}m.bin"
    save_graphs(g_save_path, [g])


