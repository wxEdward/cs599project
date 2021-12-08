import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from LSTMAttention import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
PRECISION = 5
POS_DIM_ALTER = 100

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # special linear layer for motif explainability
        self.non_linear = non_linear
        if not non_linear:
            assert(dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            #x = self.layer_norm(x)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else: # for explainability
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z

class CAWN(torch.nn.Module):
    def __init__(self, n_feat, e_feat,
                 pos_dim=0, pos_enc='spd', walk_pool='attn', walk_n_head=8, walk_mutual=False,
                 num_layers=3, drop_out=0.1, num_neighbors=20, cpu_cores=1,
                 verbosity=1, get_checkpoint_path=None, walk_linear_out=False):
        super(CAWN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = None

        # features
        self.node_features = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.edge_features = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        # dimensions of 4 elements: node, edge, time, position
        self.feat_dim = self.node_features.shape[1]  # node feature dimension
        self.edge_feat_dim = self.edge_features.shape[1]  # edge feature dimension
        self.time_dim = self.feat_dim  # default to be time feature dimension
        self.pos_dim = pos_dim  # position feature dimension
        self.pos_enc = pos_enc
        self.model_dim = self.feat_dim + self.edge_feat_dim + self.time_dim + self.pos_dim
        self.logger.info('neighbors: {}, node dim: {}, edge dim: {}, pos dim: {}, edge dim: {}'.format(self.num_neighbors, self.feat_dim, self.edge_feat_dim, self.pos_dim, self.time_dim))

        # walk-based attention/summation model hyperparameters
        self.walk_pool = walk_pool
        self.walk_n_head = walk_n_head
        self.walk_mutual = walk_mutual
        self.walk_linear_out = walk_linear_out

        # dropout for both tree and walk based model
        self.dropout_p = drop_out

        # embedding layers and encoders
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.edge_features, padding_idx=0, freeze=True)
        # self.source_edge_embed = nn.parameter(torch.tensor()self.edge_feat_dim)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.node_features, padding_idx=0, freeze=True)
        self.time_encoder = TimeEncoder(expand_dim=self.time_dim)
        self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.num_layers, ngh_finder=self.ngh_finder,
                                                cpu_cores=cpu_cores, verbosity=verbosity, logger=self.logger, enc=self.pos_enc)
        # attention model
        self.random_walk_attn_model = RandomWalkAttention(feat_dim=self.model_dim, pos_dim=self.pos_dim,
                                                     model_dim=self.model_dim, out_dim=self.feat_dim,
                                                     walk_pool=self.walk_pool,
                                                     n_head=self.walk_n_head, mutual=self.walk_mutual,
                                                     dropout_p=self.dropout_p, logger=self.logger, walk_linear_out=self.walk_linear_out)

        # final projection layer
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=not self.walk_linear_out)

        self.get_checkpoint_path = get_checkpoint_path

    def forward(self, src_idx, dst_idx, neg_samples_idx, cut_ts, edge_idx=None, test=False):
        '''
        1. grab subgraph for src, dst, neg samples
        2. add positional encoding for src & dst nodes
        3. forward propagate to get src embeddings and dst embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and neg samples embeddings (and finally neg_score (shape: [batch, ]))
        '''
        subgraph_src = self.ngh_finder.find_k_hop(self.num_layers, src_idx, cut_ts, num_neighbors=self.num_neighbors, edge_idx=edge_idx)
        subgraph_tgt = self.ngh_finder.find_k_hop(self.num_layers, src_idx, cut_ts, num_neighbors=self.num_neighbors, edge_idx=edge_idx)
        subgraph_bgd = self.ngh_finder.find_k_hop(self.num_layers, src_idx, cut_ts, num_neighbors=self.num_neighbors, edge_idx=edge_idx)


        pos_score = self.compute_score(src_idx, dst_idx, cut_ts, (subgraph_src, subgraph_tgt), test=test)
        neg_score = self.compute_score(src_idx, neg_samples_idx, cut_ts, (subgraph_src, subgraph_bgd), test=test)

        return pos_score.sigmoid(), neg_score.sigmoid()

    def compute_score(self, src_idx, dst_idx, cut_ts, subgraphs=None, test=False):
        subgraph_src, subgraph_tgt = subgraphs
        self.position_encoder.init_internal_data(src_idx, dst_idx, cut_ts, subgraph_src, subgraph_tgt)
        subgraph_src = self.subgraph_tree2walk(src_idx, cut_ts, subgraph_src)
        subgraph_tgt = self.subgraph_tree2walk(dst_idx, cut_ts, subgraph_tgt)
        src_embed = self.features_aggregation(src_idx, cut_ts, subgraph_src, test=test)
        tgt_embed = self.features_aggregation(dst_idx, cut_ts, subgraph_tgt, test=test)
        if self.walk_mutual:
            src_embed, tgt_embed = self.tune_embeddings(src_embed, tgt_embed)
        score = self.affinity_score(src_embed, tgt_embed) # score_walk shape: [B, M]
        score.squeeze_(dim=-1)

        return score

    def subgraph_tree2walk(self, src_idx, cut_ts, subgraph_src):
        # put src nodes and extracted subgraph together
        """
        subgraph_src:(node_records, eidx_records, t_records)
        node_records: (k, batch, num_neighbors ** hop_variable)
        eidx_records: (k, batch, num_neighbors ** hop_variable)
        t_records: (k, batch, num_neighbors ** hop_variable)
        Each of them is a list of k numpy arrays.
        E.g., node_records is a list of k numpy arrays, i-th array stores the i-th hop neighbors for nodes in the batch.
        """
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [src_idx[:,np.newaxis]] + node_records
        eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        t_records_tmp = [cut_ts[:,np.newaxis], 1] + t_records

        # use the list to construct a new matrix
        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_eidx_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def features_aggregation(self, src_idx, cut_ts, subgraph_src, test=False):
        node_records, eidx_records, t_records = subgraph_src
        # 1. initialize 0-layer hidden embeddings with raw node features of all hops (later with positional encodings as well)
        # 2. get time encodings for all hops
        # 3. get edge features for all in-between hops
        # 4. iterate over hidden embeddings for each layer
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_ts, t_records)  # length self.num_layers+1
        edge_features = self.retrieve_edge_features(eidx_records)  # length self.num_layers
        position_features = self.retrieve_position_features(src_idx, node_records, cut_ts, t_records,
                                                            test=test)  # length self.num_layers+1, core contribution
        # Notice that eidx_records[:, :, 1] may be all None
        # random walk branch logic:
        # 1. get the feature matrix shaped [batch, n_walk, len_walk + 1, node_dim + edge_dim + time_dim + pos_dim]
        # 2. feed the matrix forward to LSTM, then transformer, now shaped [batch, n_walk, transformer_model_dim]
        # 3. aggregate and collapse dim=1 (using set operation), now shaped [batch, out_dim]
        final_node_embeddings = self.random_walk_attn_model.forward_one_node(hidden_embeddings, time_features, edge_features,
                                                            position_features, masks)
        return final_node_embeddings

    def tune_embeddings(self, src_embed, tgt_embed):
        return self.random_walk_attn_model.mutual_query(src_embed, tgt_embed)

    def init_hidden_embeddings(self, src_idx, node_records):
        device = self.node_features.device
        node_records_th = torch.from_numpy(node_records).long().to(device)
        hidden_embeddings = self.node_raw_embed(node_records_th)  # shape [batch, n_walk, len_walk+1, node_dim]
        masks = (node_records_th != 0).sum(dim=-1).long()  # shape [batch, n_walk], here the masks means differently: it records the valid length of each walk
        return hidden_embeddings, masks

    def retrieve_time_features(self, cut_ts, t_records):
        device = self.node_features.device
        batch = len(cut_ts)
        t_records_th = torch.from_numpy(t_records).float().to(device)
        t_records_th = t_records_th.select(dim=-1, index=0).unsqueeze(dim=2) - t_records_th
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1)).view(batch, n_walk, len_walk,
                                                                             self.time_encoder.time_dim)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        #eidx_records contains the random walks of length len_walk+1, including the src node
        device = self.node_features.device
        eidx_records_th = torch.from_numpy(eidx_records).to(device)
        eidx_records_th[:, :, 0] = 0   # NOTE: this will NOT be mixed with padded 0's since those paddings are denoted by masks and will be ignored later in lstm
        edge_features = self.edge_raw_embed(eidx_records_th)  # shape [batch, n_walk, len_walk+1, edge_dim]

        return edge_features

    def retrieve_position_features(self, src_idx, node_records, cut_ts, t_records, test=False):
        start = time.time()
        encode = self.position_encoder
        if encode.enc_dim == 0:
            return None
        batch, n_walk, len_walk = node_records.shape
        node_records_r, t_records_r = node_records.reshape(batch, -1), t_records.reshape(batch, -1)

        position_features, common_nodes = encode(node_records_r, t_records_r)
        position_features = position_features.view(batch, n_walk, len_walk, self.pos_dim)

        end = time.time()

        return position_features

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder

class PositionEncoder(nn.Module):
    '''
    Note that encoding initialization and lookup is done on cpu but encoding (post) projection is on device
    '''
    def __init__(self, num_layers, enc='spd', enc_dim=2, ngh_finder=None, verbosity=1, cpu_cores=1, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc = enc
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)  # reserved for when the internal position encoding does not match input
        self.cpu_cores = cpu_cores
        self.ngh_finder = ngh_finder
        self.verbosity = verbosity
        self.logger = logger
        self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))  # landing prob at [0, 1, ... num_layers]

    def init_internal_data(self, src_idx, dst_idx, cut_ts, subgraph_src, subgraph_tgt):
        if self.enc_dim == 0:
            return
        # initialize internal data structure to index node positions
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx, dst_idx, cut_ts, subgraph_src,
                                                                subgraph_tgt)

    def collect_pos_mapping_ptree(self, src_idx, dst_idx, cut_ts, subgraph_src, subgraph_tgt):
        # Return:
        # nodetime2idx_maps: a list of dict {(node index, rounded time string) -> index in embedding look up matrix}
        if self.cpu_cores == 1:
            subgraph_src_node, _, subgraph_src_ts = subgraph_src  # only use node index and timestamp to identify a node in temporal graph
            subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
            nodetime2emb_maps = {}
            for row in range(len(src_idx)):
                src = src_idx[row]
                tgt = dst_idx[row]
                cut_time = cut_ts[row]
                src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
                src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
                tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
                tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
                nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                   src_neighbors_node, src_neighbors_ts,
                                                                   tgt_neighbors_node, tgt_neighbors_ts, batch_idx=row, enc=self.enc)
                nodetime2emb_maps.update(nodetime2emb_map)
        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src, tgt, cut_time, src_neighbors_node, src_neighbors_ts,
                                         tgt_neighbors_node, tgt_neighbors_ts, batch_idx):
        """
        This function has the potential of being written in numba by using numba.typed.Dict!
        """
        n_hop = len(src_neighbors_node)
        makekey = nodets2key
        nodetime2emb = {}
        # landing probability encoding, n_hop+1 types of probabilities for each node
        src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
        tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
        for k in range(n_hop+1): # k=0,1,...,n_hop
            k_hop_total = len(src_neighbors_node[k]) # number of k-hop neighbors of source node
            for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                          tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)

                if src_key not in nodetime2emb:
                    nodetime2emb[src_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                if tgt_key not in nodetime2emb:
                    nodetime2emb[tgt_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                nodetime2emb[src_key][0, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
                nodetime2emb[tgt_key][1, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
        null_key = makekey(batch_idx, 0, 0.0)
        nodetime2emb[null_key] = np.zeros((2, n_hop + 1), dtype=np.float32)

        return nodetime2emb

    def forward(self, node_record, t_record):
        '''
        accept two numpy arrays each of shape [batch, k-hop-support-number], corresponding to node indices and timestamps respectively
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        '''
        # encodings = []
        device = next(self.projection.parameters()).device
        # float2str = PositionEncoder.float2str
        batched_keys = make_batched_keys(node_record, t_record)
        unique, inv = np.unique(batched_keys, return_inverse=True)
        unordered_encodings = np.array([self.nodetime2emb_maps[key] for key in unique])
        encodings = unordered_encodings[inv, :]
        encodings = torch.tensor(encodings).to(device)
        common_nodes = (((encodings.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings.shape[0] * encodings.shape[1])).item()
        encodings = self.get_trainable_encodings(encodings)
        return encodings, common_nodes

    def get_trainable_encodings(self, encodings):
        '''
        Args:
            encodings: a device tensor of shape [batch, support_n, 2] / [batch, support_n, 2, L+1]
        Returns:  a device tensor of shape [batch, pos_dim]
        '''
        if self.enc == 'spd':
            encodings[encodings > (self.num_layers+0.5)] = self.num_layers + 1
            encodings = self.trainable_embedding(encodings.long())  # now shape [batch, support_n, 2, pos_dim]
            encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
        elif self.enc == 'lp':
            encodings = self.trainable_embedding(encodings.float())   # now shape [batch, support_n, 2, pos_dim]
            encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
        else:
            assert(self.enc == 'saw')
            encodings = self.trainable_embedding(encodings.float())  # now shape [batch, support_n, pos_dim]
        return encodings

class TimeEncoder(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncoder, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic

class RandomWalkAttention(nn.Module):
    '''
    RandomWalkAttention have two modules: lstm + tranformer-self-attention
    '''
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, walk_pool='attn', mutual=False, n_head=8, dropout_p=0.1, walk_linear_out=False):
        '''
        masked flags whether or not use only valid temporal walks instead of full walks including null nodes
        '''
        super(RandomWalkAttention, self).__init__()
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2  # half the model dim to save computation cost for attention
        self.out_dim = out_dim
        self.walk_pool = walk_pool
        self.mutual = mutual
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p)  # encode all types of features along each temporal walk
        self.position_encoder = FeatureEncoder(self.pos_dim, self.pos_dim, self.dropout_p)  # encode specifially spatio-temporal features along each temporal walk
        self.projector = nn.Sequential(nn.Linear(self.feature_encoder.model_dim+self.position_encoder.model_dim, self.attn_dim),  # notice that self.feature_encoder.model_dim may not be exactly self.model_dim is its not even number because of the usage of bi-lstm
                                       nn.ReLU(), nn.Dropout(self.dropout_p))
        self.self_attention = TransformerEncoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                      dim_feedforward=4*self.attn_dim, dropout=self.dropout_p,
                                                      activation='relu')
        if self.mutual:
            self.mutual_attention_src2tgt = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
            self.mutual_attention_tgt2src = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p, walk_linear_out=walk_linear_out)
        self.logger.info('bi-lstm actual encoding dim: {} + {}, attention dim: {}, attention heads: {}'.format(self.feature_encoder.model_dim, self.position_encoder.model_dim, self.attn_dim, self.n_head))

    def forward_one_node(self, hidden_embeddings, time_features, edge_features, position_features, masks=None):
        '''
        Input shape [batch, n_walk, len_walk, *_dim]
        Return shape [batch, n_walk, feat_dim]
        '''
        combined_features = self.aggregate(hidden_embeddings, time_features, edge_features, position_features)
        combined_features = self.feature_encoder(combined_features, masks)
        if self.pos_dim > 0:
            position_features = self.position_encoder(position_features, masks)
            combined_features = torch.cat([combined_features, position_features], dim=-1)
        X = self.projector(combined_features)
        if self.walk_pool == 'sum':
            X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
            return X
        else:
            X = self.self_attention(X)
            if not self.mutual:
                X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
            return X

    def mutual_query(self, src_embed, tgt_embed):
        '''
        Input shape: [batch, n_walk, feat_dim]
        '''
        src_emb = self.mutual_attention_src2tgt(src_embed, tgt_embed)
        tgt_emb = self.mutual_attention_tgt2src(tgt_embed, src_embed)
        src_emb = self.pooler(src_emb)
        tgt_emb = self.pooler(tgt_emb)
        return src_emb, tgt_emb

    def aggregate(self, hidden_embeddings, time_features, edge_features, position_features):
        batch, n_walk, len_walk, _ = hidden_embeddings.shape
        device = hidden_embeddings.device
        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features, position_features], dim=-1)
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features


class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_p=0.1):
        super(FeatureEncoder, self).__init__()
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        self.lstm_encoder = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch*n_walk, len_walk, feat_dim)
        if mask is not None:
            lengths = mask.view(batch*n_walk)
            X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        encoded_features = self.lstm_encoder(X)[0]
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = encoded_features.select(dim=1, index=-1).view(batch, n_walk, self.model_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

class SetPooler(nn.Module):
    """
    Implement similar ideas to the Deep Set
    """
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if self.walk_linear_out:  # for explainability, postpone summation to merger function
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        else:
            assert(agg == 'mean')
            return self.out_proj(X.mean(dim=-2))

