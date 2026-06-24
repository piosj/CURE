import torch
import numpy as np
from torch import nn
from typing import Optional, List

from utils.nce_loss import NCELoss
from utils.MSA_news_encoder import NewsEncoder


class TUDOR(nn.Module):
    """
    News Encoder
    initialize news embeddings
    
    LightGCN
    
    GRNN
    only for user:
    1) define LSTM
    2) conduct LSTM 
    
    Loss function
    1) compute scores
    2) define NLL loss 
    3) train NLL loss
    
    forward
    1) load graph informations
    2) conduct LightGCN
    3) conduct GRNN
    4) compute Loss
    """
    
    def __init__(self, all_news_ids, news_id_to_info, user_num, cat_num, news_num, pretrained_word_embedding=None, emb_dim=100, batch_size=300, snapshots_num=1680, config=None):
        super(TUDOR, self).__init__()
        self.config = config
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.snapshots_num = snapshots_num
        self.device = torch.device(f"cuda:{config.gpu_num}" if torch.cuda.is_available() else "cpu")
        self.user_embedding_layer = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN user
        self.cat_embedding_layer = nn.Embedding(num_embeddings=cat_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN relation
        # Variables for GRNN
        self.user_num = user_num
        self.cat_num = cat_num
        self.news_num = news_num
        self.unique_category = config.unique_category
        self.c0_embedding_layer_u = nn.Embedding(num_embeddings=user_num+news_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # for cell state in LSTM
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device)   # input dim, hn dim
        # Variables for News_Encoder
        self.pretrained_word_embedding = pretrained_word_embedding
        self.news_encoder = NewsEncoder(self.config, self.pretrained_word_embedding).to(self.device)
        self.all_news_ids = all_news_ids   
        self.news_id_to_info = news_id_to_info
        
    def News_Encoder(self, news_ids, max_batch: int = 512):
        """
        news_ids  : list(int) of news_int
        max_batch : batch size
        """
        news_embeddings = torch.zeros((len(news_ids), self.emb_dim)).to(self.device)   # (# of news, news embedding dim)

        batch_titles, batch_cats, batch_scats, batch_idx = [], [], [], []

        def _flush():
            assert all(0 <= idx < len(news_ids) for idx in batch_idx), \
                f"Exceed batch_idx range! max={max(batch_idx)}, news_ids len={len(news_ids)}"

            if not batch_titles:
                return

            padded = nn.utils.rnn.pad_sequence(
                [torch.tensor(t, dtype=torch.long) for t in batch_titles],
                batch_first=True, padding_value=0
            ).to(self.device)

            cats = torch.tensor(batch_cats, dtype=torch.long, device=self.device)
            scats = torch.tensor(batch_scats, dtype=torch.long, device=self.device)

            nv = self.news_encoder(padded, cats, scats)   # (B, emb_dim)
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
    
    # Initialize LightGCN (K, alpha)
    def _init_lightgcn(self, K: int = 3, alpha: Optional[List[float]] = None):
        """K: propagation layer num, alpha: layer combination weight list"""
        self.K = K
        
        if alpha is None:
            self.alpha = torch.tensor([1.0 / (K + 1)] * (K + 1), device=self.device)
        else:
            assert len(alpha) == K + 1
            self.alpha = torch.tensor(alpha, device=self.device)
    
    def lgcn_message_func(self, edges):
        coefficient = torch.rsqrt(edges.src['deg'] * edges.dst['deg']).unsqueeze(1)
        cat_embedding = self.rel_embedding[edges.data['cat_idx'].type(torch.LongTensor).to(self.device)]

        msg = edges.src['h'] * cat_embedding * coefficient
        return {'msg': msg}
    
    def lgcn_reduce_func(self, nodes):
        return {'h': nodes.mailbox['msg'].sum(1)}
    
    def propagate_lightgcn(self, g, seed_nodes, edges):
        g.ndata['deg'] = g.in_degrees().float().clamp(min=1).to(self.device)
        g.ndata['h'] = g.ndata['node_emb']

        h_accum = self.alpha[0] * g.ndata['h']          
        for k in range(self.K):
            g.send_and_recv(edges=edges)
            h_accum = h_accum + self.alpha[k+1] * g.ndata['h']  

        g.ndata['node_emb'] = h_accum

        with torch.no_grad():
            del g.ndata['deg'], g.ndata['h'];  torch.cuda.empty_cache()
        return h_accum[seed_nodes]


    def seq_GCRNN_batch(self, g, sub_g, latest_train_time, seed_list, history_length):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.news_num)
        for i in range(latest_train_time, -1, -1): # latest -> 0; start from latest time
            
            # It needs to be saved user and news indicies of seed_list in order
            # news indicies: user_num + news_int
            check_lifetime[list(seed_list[i])] = history_length # seed_list: seed users splitted by time idx

            # Add users of seed list to future needed nodes
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            
            # 1hop edges of seed at i
            hop1_u, hop1_v = sub_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')
            # u (source) -> v (desination)
            
            gcn_seed_per_time.append(list(future_needed_nodes))   # Seed            
            
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))   # 1-hop bi-directed edges
            
            
            check_lifetime[check_lifetime>0] -= 1   # Applying history length
            try:
                future_needed_nodes = future_needed_nodes - set(np.where(check_lifetime==0)[0]) # seed next
            except:
                pass
        
        self.rel_embedding = self.cat_embedding_layer(torch.tensor(range(self.cat_num)).to(self.device))
        
        user_embeddings = self.user_embedding_layer(torch.tensor(range(self.user_num)).to(self.device))
        news_embeddings = self.News_Encoder(self.all_news_ids)

        g.ndata['node_emb'] = torch.cat([user_embeddings, news_embeddings], dim=0)
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.arange(self.user_num+self.news_num, device=self.device))

        entity_embs = []
        entity_index = []
        
        # Init LightGCN
        self._init_lightgcn(K=self.config.hop)  
        
        # If you use dgl after 0.9, you need to replace 'register function' to that version's form
        g.register_message_func(self.lgcn_message_func)
        g.register_reduce_func(self.lgcn_reduce_func)
        for i in range(latest_train_time+1): # 0 -> latest
            inverse = latest_train_time - i   # snapshots-num - i
            # gcn_seed_per_time -> start from latest 
            if len(gcn_seed_per_time[inverse]) > 0:   
                changed = sorted(gcn_seed_per_time[inverse])   # Make seed user list of that time by user_id sorted 

                user_seed_ = changed   
                
                user_prev_hn = g.ndata['node_emb'][user_seed_]
                user_prev_cn = g.ndata['cx'][user_seed_]

                user_input = self.propagate_lightgcn(g, user_seed_, gcn_seed_1hopedge_per_time[inverse])
                
                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))   
                
                old_emb = g.ndata['node_emb']                 # present node embeddings (tensor)
                new_emb = old_emb.clone()                     
                new_emb[user_seed_] = user_hn                 # update seeds
                g.ndata['node_emb'] = new_emb                 # assign new embeddings

                # Same process for LSTM cell state
                old_cx = g.ndata['cx']
                new_cx = old_cx.clone()
                new_cx[user_seed_] = user_cn
                g.ndata['cx'] = new_cx
                
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]   
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i   # Every user has ideal index by timestamps
                entity_embs.append(seed_emb)   
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))

        entity_embs = torch.cat(entity_embs).to(self.device)   

        # entity_index before torch.cat: len=100(elements: [0], [0], ..., [batch size]), after: shape=(batch size,)
        entity_index = torch.cat(entity_index)   
        # shape: (Sum of users exist every snapshot, )
        
        ent_embs = entity_embs[entity_index.argsort()]   # Sorted by user indicies
        
        return ent_embs
        
    
    def forward(self, user_batch, news_batch, category_batch, time_batch, g, sub_g, ns_idx, history_length=100): 
        """
        g
           - edges = ('user', 'news'),
                     ('news', 'user')
        """
        
        seed_list = []
        seed_entid = []
        train_t = []
        latest_train_time = self.snapshots_num - 1
        for _ in range(latest_train_time+1):
            seed_list.append(set())
            
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                train_t.append(time)
                
        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), 
                                            sorted = True, return_inverse = True)
        
        user_embs = ent_embs[index_for_ent_emb]   # (train_click_num, emb_dim)
                                                  
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # candidate_n_embs: (train_click_num, (1 + 4), emb_dim); 1: target, 4: # of ns samples
        # ns_idx: (train_click_num, 5)
        candidate_user_embs = user_embs # (train_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (train_click_num, 1, emb_dim)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)   # (train_click_num, 5)
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
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
        
        latest_train_time = self.snapshots_num - 1   
        seed_entid = []
        test_t = []
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                test_t.append(time)

        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)   # (batch_size, emb_dim)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(test_t), 
                                            sorted = True, return_inverse = True)
        # (batch_size, )
        u_time_embs = ent_embs[index_for_ent_emb]

        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # candidate_n_embs: (test_click_num, (1 + 20), emb_dim); 1: target, 20: # of ns samples
        # ns_idx: (test_click_num, 21)
        candidate_user_embs = u_time_embs#[user_score_idx]   # user_score_idx: (test_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (test_click_num, 1, emb_dim)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # candidate_n_embs: (test_click_num, emb_dim)*(test_click_num, 21, emb_dim)
        # candidate_score: (test_click_num, 21)        
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   
        
        return candidate_score, loss
