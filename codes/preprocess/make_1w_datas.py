import pandas as pd
import numpy as np
import os

"""
Due restriction of data capacity(50MB), it is impossible to add 'behaviors.tsv' which contains 7w data.
Thus, I add '1w_behaviors.tsv' and commented out the below section that generates '1w_behaviors.tsv'.
"""

### Extract 1w behaviors from 7w data(behaviors.tsv)

# # Load behaviors.tsv which has same format of file '1w_behaviors.tsv' 
# df = pd.read_csv("data/behaviors.tsv", sep='\t', parse_dates=['click_time'])

# start_time = pd.Timestamp("2017-01-05 00:00:00")
# end_time = pd.Timestamp("2017-01-11 23:59:59")
# df_filtered = df[(df['click_time'] >= start_time) & (df['click_time'] <= end_time)].copy()

# unique_users_all = df_filtered['history_user'].nunique()

# # Split periods (history train test)
# df_filtered['date'] = df_filtered['click_time'].dt.date
# period_1 = df_filtered[(df_filtered['date'] >= pd.to_datetime('2017-01-05').date()) &   # 5d
#                        (df_filtered['date'] <= pd.to_datetime('2017-01-09').date())]
# period_2 = df_filtered[(df_filtered['date'] == pd.to_datetime('2017-01-10').date())]    # 1d
# period_3 = df_filtered[(df_filtered['date'] == pd.to_datetime('2017-01-11').date())]    # 1d

# users_p1 = set(period_1['history_user'].unique())
# users_p2 = set(period_2['history_user'].unique())
# users_p3 = set(period_3['history_user'].unique())

# common_users = users_p1 & users_p2 & users_p3

# # Filtering users who have at least a click every periods
# df_common = df_filtered[df_filtered['history_user'].isin(common_users)]
# common_user_count = df_common['history_user'].nunique()
# common_news_count = df_common['clicked_news'].nunique()
# common_click_count = len(df_common)
# print(f"user num: {common_user_count}, news num: {common_news_count}, click num: {common_click_count}")

# # Filtering user who have at least 10 clicks over three periods
# user_click_counts = df_common.groupby('history_user').size()
# qualified_users = user_click_counts[user_click_counts >= 10].index

# df_final = df_common[df_common['history_user'].isin(qualified_users)]

# final_user_count = df_final['history_user'].nunique()
# final_news_count = df_final['clicked_news'].nunique()
# final_click_count = len(df_final)

# print(f"final user num: {final_user_count}")
# print(f"final news num: {final_news_count}")
# print(f"final click num: {final_click_count}")

# train_ = df_final[(df_final['date'] == pd.to_datetime('2017-01-10').date())]
# test_ = df_final[(df_final['date'] == pd.to_datetime('2017-01-11').date())]
# print(f"final train/test click num: {len(train_['history_user'])} / {len(test_['history_user'])}")


# df_final = df_final.drop(columns=['date'])

# df_final[['category', 'subcategory']] = df_final['category'].str.split('|', n=1, expand=True)
# cols_order = ['history_user', 'clicked_news', 'click_time',
#               'category', 'subcategory', 'title']
# df_final = df_final[cols_order]

# # save 1w_behaviors.tsv
# folder_path = 'Adressa_1w/datas'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
# df_final.to_csv("Adressa_1w/datas/1w_behaviors.tsv", sep='\t', index=False)



### Load 1w behaviors file.
df_final = pd.read_csv('Adressa_1w/datas/1w_behaviors.tsv', sep='\t')

### Preprocessing for 1w datas
# news2int.tsv
unique_news = df_final['clicked_news'].unique()
news2int = {news: i for i, news in enumerate(unique_news)}
news2int_df = pd.DataFrame(list(news2int.items()), columns=['news_id', 'news_int'])
news2int_df.to_csv("Adressa_1w/datas/news2int.tsv", sep='\t', index=False)


# user2int.tsv
unique_users = df_final['history_user'].unique()
user2int = {user: i for i, user in enumerate(unique_users)}
user2int_df = pd.DataFrame(list(user2int.items()), columns=['user_id', 'user_int'])
user2int_df.to_csv("Adressa_1w/datas/user2int.tsv", sep='\t', index=False)


### all_news.tsv
# 	clicked_news	category	subcategory	title
all_news = pd.read_csv('data/all_news.tsv', sep='\t')
all_news_1w = all_news[all_news['clicked_news'].isin(set(unique_news))]
all_news_1w.to_csv("Adressa_1w/datas/all_news.tsv", sep='\t', index=False)


# all_news_nyheter_splitted.tsv
nyheter_mask = all_news['category'] == 'nyheter'
all_news_ns = all_news.copy()
all_news_ns.loc[nyheter_mask, 'category'] = all_news_ns.loc[nyheter_mask, 'subcategory']
all_news_ns.loc[nyheter_mask, 'subcategory'] = 'No subcategory'
all_news_ns.loc[nyheter_mask, 'category'] = (
    all_news_ns.loc[nyheter_mask, 'category'] + '_nyheter'
)
all_news_ns.to_csv('data/all_news_nyheter_splitted.tsv', sep='\t', index=False)

all_news_1w_ns = all_news_ns[all_news_ns['clicked_news'].isin(set(unique_news))]
all_news_1w_ns.to_csv("Adressa_1w/datas/all_news_nyheter_splitted.tsv", sep='\t', index=False)


# category2int.tsv
categories = all_news_1w['category'].unique().tolist()#[1:]
subcategories = all_news_1w['subcategory'].unique().tolist()

print("length of categories:", len(categories))  
print("length of subcategories:", len(subcategories))  

categories.remove('No category')
subcategories.remove('No subcategory')

cat2int = {
    'No category': 0,
    'No subcategory': 0
}
idx = 1
for cat in categories + subcategories:
    cat2int[cat] = idx
    idx += 1

category2int = pd.DataFrame(
    list(cat2int.items()),
    columns=['category', 'int']
)

category2int.to_csv(
    "Adressa_1w/datas/category2int.tsv",
    sep='\t', index=False
)


# category2int_nyheter_splitted.tsv
categories = all_news_1w_ns['category'].unique().tolist()#[1:]
categories.remove('No category')
categories = ['No category'] + categories
print("length of categories_nyheter_splitted:", len(categories))   

cat2int_ns = {cat: i for i, cat in enumerate(categories)}
category2int_ns = pd.DataFrame(list(cat2int_ns.items()), columns=['category', 'int'])
category2int_ns.to_csv("Adressa_1w/datas/category2int_nyheter_splitted.tsv", sep='\t', index=False)
