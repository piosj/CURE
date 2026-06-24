import pandas as pd


train_news_file_path = 'Adressa_1w/datas/all_news_nyheter_splitted.tsv'
train_news_df = pd.read_csv(train_news_file_path, sep='\t')
train_news_df.columns = ['index_col', 'newsId','category','subcategory', 'title']
sub_train_news_df = train_news_df[['newsId', 'category']]

train_ns_path = "Adressa_1w/train/train_negative_samples_lt36_ns4.tsv"
train_ns = pd.read_csv(train_ns_path, sep='\t')

train_file_path = 'Adressa_1w/datas/1w_behaviors.tsv'
df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]

train_df = train_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')

train_users = train_df['history_user']

train_df = train_df.dropna(subset=['clicked_news'])
train_df = train_df[train_df.notna()]



news2int_file_path = 'Adressa_1w/datas/news2int.tsv'
news2int = pd.read_csv(news2int_file_path, sep='\t')

news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))

user2int_df = pd.read_csv('Adressa_1w/datas/user2int.tsv', sep='\t')
user2int = dict(zip(user2int_df['user_id'], user2int_df['user_int']))


# Adding 'news_int' and 'negative samples' to train_df
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


train_ns.to_csv('Adressa_1w/train/train_ns.tsv', sep='\t', index=False)