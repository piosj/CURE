import pickle
import pandas as pd
import torch
from tqdm import tqdm
import datetime
import os

def make_train_datas(interval_minutes):
    news2int_file_path = 'news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    train_file_path = f'behaviors.tsv'
    df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')

    df['click_time'] = pd.to_datetime(df['click_time'])
    
    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-23 00:00:00')
    train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    train_df = train_df.dropna(subset=['clicked_news'])

    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
    user2int_df = pd.read_csv('user2int.tsv', sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    all_user_ids = user2int_df['user_int'].tolist()   

    train_df['user_int'] = train_df['history_user'].map(user2int)
    train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)
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

    train_df['cat_int'] = train_df.apply(get_cat_int, axis=1)

    
    GLOBAL_START = pd.Timestamp('2017-01-05 00:00:00')  
    def get_period_start_global(click_time, interval_minutes):
        delta = click_time - GLOBAL_START
        periods = int(delta.total_seconds() // (interval_minutes * 60))
        return GLOBAL_START + pd.Timedelta(minutes=interval_minutes * periods)


    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['Period_Start'] = train_df['click_time'].apply(lambda x: get_period_start_global(x, interval_minutes=interval_minutes))
    
    history_weeks = 15/7
    interval_hours = interval_minutes / 60
    his_snapshots_num = int(history_weeks * 7 * 24 / interval_hours)
    
    unique_period_starts = train_df['Period_Start'].unique()
    time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
    train_df['time_idx'] = train_df['Period_Start'].map(time_dict)
    train_df = train_df[train_df['time_idx'] >= his_snapshots_num]

    train_news = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(train_df[train_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        train_news.append(u_news)
        
    train_category = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_category = torch.tensor(train_df[train_df['user_int'] == u_id]['cat_int'].values, dtype=torch.long)
        train_category.append(u_category)

    train_time = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_time = torch.tensor(train_df[train_df['user_int'] == u_id]['time_idx'].values, dtype=torch.long)
        train_time.append(u_time)
    
    return list(zip(train_news, train_category, train_time))

