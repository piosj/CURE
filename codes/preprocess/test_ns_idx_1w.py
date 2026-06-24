import pandas as pd
import numpy as np

"""
making test_ns.tsv (train_negative_samples_lt36_ns4 -> train_ns; mapping news to int)

"""


# news category
train_news_file_path = 'Adressa_1w/datas/all_news_nyheter_splitted.tsv'
train_news_df = pd.read_csv(train_news_file_path, sep='\t')
train_news_df.columns = ['index_col', 'newsId','category','subcategory', 'title']
sub_train_news_df = train_news_df[['newsId', 'category']]

# train_ns data
train_ns_path = "Adressa_1w/test/test_negative_samples_lt36_ns20.tsv"
train_ns = pd.read_csv(train_ns_path, sep='\t')

# Loading 1w data
train_file_path = 'Adressa_1w/datas/1w_behaviors.tsv'
df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]

train_df = train_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')
train_df = train_df.dropna(subset=['clicked_news'])
train_df = train_df[train_df.notna()]

train_users = train_df['history_user']


# news2int mapping
news2int_file_path = 'Adressa_1w/datas/news2int.tsv'
news2int = pd.read_csv(news2int_file_path, sep='\t')

news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
user2int_df = pd.read_csv('Adressa_1w/datas/user2int.tsv', sep='\t')
user2int = dict(zip(user2int_df['user_id'], user2int_df['user_int']))

# mapping negative samples to news int
train_ns['news_int'] = train_ns['clicked_news'].map(news2int_mapping)

def map_negative_samples(ns_str):
    if pd.isna(ns_str):
        return ns_str
    news_ids = ns_str.split()
    news_ints = [str(news2int_mapping.get(nid, -1)) for nid in news_ids]
    return " ".join(news_ints)

train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
train_ns['user_int'] = train_ns['user'].map(user2int)
train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)


# ===== Calculate remaining lifetime weight =====
pub_path = "Adressa_1w/datas/news_publish_times.tsv"
pub_df   = pd.read_csv(pub_path, sep='\t', usecols=['news_id', 'publish_time'])

pub_df['publish_time'] = pd.to_datetime(pub_df['publish_time'])
pub_df['news_int'] = pub_df['news_id'].map(news2int_mapping)
pub_dict = dict(zip(pub_df['news_int'], pub_df['publish_time']))  

train_ns['click_time'] = pd.to_datetime(train_ns['click_time'])
train_ns['publish_times_list'] = (
    train_ns['publish_times']
    .str.split(',')                                    # ["2017-01-22 05:07:45", ...]
    .apply(lambda lst: pd.to_datetime(lst))  
)

train_ns['pos_publish_time'] = train_ns['news_int'].map(pub_dict)


THRESHOLD_HOUR = 36 
ALPHA = 0.1              # 0.2
LIFETIME_MIN = THRESHOLD_HOUR #* 60     # 36h

def weight_by_lifetime(click_t, pub_t):
    remaining = LIFETIME_MIN - abs((click_t - pub_t).total_seconds()) / 60 / 60  
    return 1 / (1 + np.exp(-ALPHA * remaining))     

def compute_row_weight(row):
    click_t = row['click_time']
    weights = []

    # positive
    pos_pub = row['pos_publish_time']
    weights.append(f'{weight_by_lifetime(click_t, pos_pub):.6f}')

    # negatives
    for neg_pub_t in row['publish_times_list']:
        weights.append(f'{weight_by_lifetime(click_t, neg_pub_t):.6f}')

    return ' '.join(weights)


train_ns['candidate_weight'] = train_ns.apply(compute_row_weight, axis=1)

train_ns.drop(columns=['pos_publish_time', 'publish_times_list'], inplace=True)

# Save train_ns.tsv
train_ns.to_csv('Adressa_1w/test/test_ns.tsv', sep='\t', index=False)