from torch import nn
import dgl
import torch
from tqdm import tqdm
import numpy as np
import gc
import random
from utils.nce_loss import NCELoss
from model.config import Config
from utils.MSA_news_encoder import NewsEncoder


torch.cuda.set_device(Config.gpu_num)
random_seed = Config.seed
random.seed(random_seed)
torch.manual_seed(random_seed)

class GCRNN(nn.Module):
    def __init__(self, all_news_ids, news_id_to_info, user_num, cat_num, news_num, pretrained_word_embedding, emb_dim, batch_size, snapshots_num):
        super(GCRNN, self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.snapshots_num = snapshots_num
        self.device = torch.device(f"cuda:{Config.gpu_num}" if torch.cuda.is_available() else "cpu")
        self.user_embedding_layer = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   
        self.cat_embedding_layer = nn.Embedding(num_embeddings=cat_num, embedding_dim=emb_dim, sparse = False).to(self.device)   
        self.user_num = user_num
        self.cat_num = cat_num
        self.news_num = news_num
        self.c0_embedding_layer_u = nn.Embedding(num_embeddings=user_num+news_num, embedding_dim=emb_dim, sparse = False).to(self.device)   
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device)   
        
        self.config = Config
        self.pretrained_word_embedding = pretrained_word_embedding
        self.news_encoder = NewsEncoder(self.config, self.pretrained_word_embedding).to(self.device)
        self.all_news_ids = all_news_ids   
        self.news_id_to_info = news_id_to_info
        
    def News_Encoder(self, news_ids, max_batch: int = 512):
        news_embeddings = torch.zeros((len(news_ids), self.emb_dim)).to(self.device)  
        batch_titles, batch_cats, batch_scats, batch_idx = [], [], [], []

        def _flush():
            if not batch_titles:
                return

            padded = nn.utils.rnn.pad_sequence(
                [torch.tensor(t, dtype=torch.long) for t in batch_titles],
                batch_first=True, padding_value=0
            ).to(self.device)

            cats = torch.tensor(batch_cats, dtype=torch.long, device=self.device)
            scats = torch.tensor(batch_scats, dtype=torch.long, device=self.device)

            nv = self.news_encoder(padded, cats, scats) 
            news_embeddings[batch_idx] = nv

            batch_titles.clear(); batch_cats.clear()
            batch_scats.clear(); batch_idx.clear()

        for i, nid in enumerate(news_ids):
            if nid in self.news_id_to_info:
                info = self.news_id_to_info[nid]
                batch_titles.append(info['title_idx'])
                batch_cats.append(info['category_idx'])
                batch_scats.append(info['subcategory_idx'])
                batch_idx.append(i)
            else:                                      
                news_embeddings[i] = torch.randn(self.emb_dim, device=self.device)
           
            if len(batch_titles) == max_batch:
                _flush()
        _flush()

        return news_embeddings

    
    def message_func(self, edges):
        return {'msg': edges.src['node_emb'] * self.rel_embedding[edges.data['cat_idx'].type(torch.LongTensor)]}

    def reduce_func(self, nodes):
        aggregated = nodes.mailbox['msg'].mean(1)
        return {'node_emb2': aggregated}        

    def seq_GCRNN_batch(self, g, sub_g, latest_train_time, seed_list, history_length):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        gcn_1hopneighbor_per_time = []
        gcn_seed_2hopedge_per_time = []
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num)
        for i in range(latest_train_time, -1, -1): 
            check_lifetime[list(seed_list[i])] = history_length
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            
            hop1_u, hop1_v = sub_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')
            gcn_seed_per_time.append(list(future_needed_nodes)) 
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))

            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - set(np.where(check_lifetime==0)[0])
            except:
                pass
        
        self.rel_embedding = self.cat_embedding_layer(torch.tensor(range(self.cat_num)).to(self.device))
        g.ndata['node_emb'] = torch.zeros(g.number_of_nodes(), self.emb_dim, device=self.device)
        g.ndata['node_emb'][:self.user_num] = self.user_embedding_layer(torch.tensor(range(self.user_num)).to(self.device))
        g.ndata['node_emb'][self.user_num:] = self.News_Encoder(self.all_news_ids)
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device))
        
        entity_embs = []
        entity_index = []
        g.register_message_func(self.message_func)
        g.register_reduce_func(self.reduce_func)
        for i in range(latest_train_time+1):
            inverse = latest_train_time - i  
            if len(gcn_seed_per_time[inverse]) > 0:   
                changed = sorted(gcn_seed_per_time[inverse])   

                user_seed_ = changed   
                user_prev_hn = g.ndata['node_emb'][user_seed_]
                user_prev_cn = g.ndata['cx'][user_seed_]

                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse])
                if edge_num > 0:
                    try:
                        g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                        g.ndata.pop('node_emb2')
                    except:
                        pass
                user_input = g.ndata['node_emb'][user_seed_]

                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))
                g.ndata['node_emb'][user_seed_] = user_hn
                g.ndata['cx'][user_seed_] = user_cn
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]  
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i   
                entity_embs.append(seed_emb)   
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))

        entity_embs = torch.cat(entity_embs).to(self.device)   
        entity_index = torch.cat(entity_index)   
        ent_embs = entity_embs[entity_index.argsort()]  
        
        return ent_embs
        
    
    def forward(self, user_batch, news_batch, category_batch, time_batch, g, sub_g, ns_idx, history_length=100):        
        seed_list = []
        seed_entid = []
        train_t = []
        latest_train_time = self.snapshots_num - 1
        for i in range(latest_train_time+1):
            seed_list.append(set())

        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                try:
                    seed_list[time].add(user)  
                    seed_entid.append(user)
                    train_t.append(time)
                except:
                    print("time:", time)
                    exit()
                    
                
        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), 
                                            sorted = True, return_inverse = True)
        
        user_embs = ent_embs[index_for_ent_emb]  
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        candidate_user_embs = user_embs
        candidate_user_embs = candidate_user_embs.unsqueeze(1)           
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)  
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   

        return loss
    
    
    def inference(self, user_batch, news_batch, time_batch, g, sub_g, ns_idx, history_length=100):
        seed_list = []
        seed_entid = []
        test_t = []
        
        for time_list in time_batch:
            for time in time_list:
                test_t.append(time)
        
        latest_train_time = self.snapshots_num-1
        seed_entid = []
        test_t = []
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                test_t.append(time)

        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)  
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(test_t), 
                                            sorted = True, return_inverse = True)
        u_time_embs = ent_embs[index_for_ent_emb]
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        candidate_user_embs = u_time_embs
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   
        
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
      
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)  
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   
        
        return candidate_score, loss

