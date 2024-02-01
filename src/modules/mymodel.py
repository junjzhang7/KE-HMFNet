import logging
import math

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .gnn import GNNGraph
from .SetTransformer import SAB

logger = logging.getLogger(__name__)


class PatientEncoder(nn.Module):
    def __init__(self, voc_size, hidden_size, dropout=0.1, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.med_voc_size = voc_size[2]
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(voc_size[0], hidden_size),
                torch.nn.Embedding(voc_size[1], hidden_size),
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

        self.rnn_encoders = torch.nn.ModuleList(
            [
                torch.nn.GRU(hidden_size, hidden_size, batch_first=True),
                torch.nn.GRU(hidden_size, hidden_size, batch_first=True),
            ]
        )

        self.query = torch.nn.Sequential(
            torch.nn.ReLU(), nn.Linear(hidden_size * 2, hidden_size)
        )

        self.init_weights()

    def forward(self, patient_record):
        seq_emb_diag, seq_emb_proc = [], []
        for admission in patient_record:
            Idx1 = torch.LongTensor([admission[0]]).to(self.device)
            Idx2 = torch.LongTensor([admission[1]]).to(self.device)

            diag_emb = self.dropout(self.embeddings[0](Idx1))
            proc_emb = self.dropout(self.embeddings[1](Idx2))

            seq_emb_diag.append(torch.sum(diag_emb, keepdim=True, dim=1))
            seq_emb_proc.append(torch.sum(proc_emb, keepdim=True, dim=1))

        seq_emb_diag = torch.cat(seq_emb_diag, dim=1)
        seq_emb_proc = torch.cat(seq_emb_proc, dim=1)

        rnn_outs_diag, hidden_diag = self.rnn_encoders[0](seq_emb_diag)
        rnn_outs_proc, hidden_proc = self.rnn_encoders[1](seq_emb_proc)

        visit_feats = torch.cat([rnn_outs_diag, rnn_outs_proc], dim=-1)

        queries = self.query(visit_feats).squeeze(0)
        query = queries[-1:]

        if len(patient_record) > 1:
            history_keys = queries[:-1]

            history_values = np.zeros((len(patient_record) - 1, self.med_voc_size))
            for idx, admission in enumerate(patient_record):
                if idx == len(patient_record) - 1:
                    break
                history_values[idx, admission[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)
        else:
            history_keys = None
            history_values = None

        return query, history_keys, history_values

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class PriorEncoder(nn.Module):
    def __init__(
        self, med_voc_size, ehr_adj_path, ddi_adj_path, emb_dim, device
    ) -> None:
        super().__init__()
        ehr_adj = dill.load(open(ehr_adj_path, "rb"))
        ddi_adj = dill.load(open(ddi_adj_path, "rb"))

        self.ehr_gcn = GCN(
            voc_size=med_voc_size, emb_dim=emb_dim, adj=ehr_adj, device=device
        )
        self.ddi_gcn = GCN(
            voc_size=med_voc_size, emb_dim=emb_dim, adj=ddi_adj, device=device
        )

        self.inter = nn.Parameter(torch.FloatTensor(1)).to(device)
        nn.init.normal_(self.inter, 0, 1 / emb_dim)

        self.proj = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, query, history_keys, history_values):
        prior_graph_emb = self.ehr_gcn() - self.inter * self.ddi_gcn()  # [131, 64]
        # 1. read drug_graph memory
        weights_embedding = F.softmax(torch.mm(query, prior_graph_emb.t()), dim=-1)
        weights_embedding = torch.diag(weights_embedding.squeeze(0))
        graph_context1 = torch.matmul(weights_embedding, prior_graph_emb)
        # 2. read history memory
        if history_keys is None:
            graph_context2 = graph_context1
        else:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)
            weighted_values = visit_weight.mm(history_values)

            weighted_values = torch.diag(weighted_values.squeeze(0))

            graph_context2 = torch.mm(weighted_values, prior_graph_emb)

        memory_context = self.proj(torch.cat([graph_context1, graph_context2], -1))

        return memory_context


class MyModel(torch.nn.Module):
    def __init__(
        self,
        args,
        emb_dim,
        voc_size,
        substruct_num,
        ehr_adj_path,
        ddi_adj_path,
        device=torch.device("cuda"),
        dropout=0.5,
    ):
        super().__init__()
        self.best_model_path = None
        self.args = args
        self.device = device
        self.med_voc_size = voc_size[2]
        self.emb_dim = emb_dim

        self.patient_encoder = PatientEncoder(voc_size, emb_dim, dropout, device)

        self.prior_encoder = PriorEncoder(
            self.med_voc_size, ehr_adj_path, ddi_adj_path, emb_dim, device
        )

        self.global_encoder = GNNGraph(
            num_layer=4,
            emb_dim=self.emb_dim,
            gnn_type="gin",
            virtual_node=False,
            drop_ratio=dropout,
        )
        self.substruct_encoder = GNNGraph(
            num_layer=4,
            emb_dim=self.emb_dim,
            gnn_type="gin",
            virtual_node=False,
            drop_ratio=dropout,
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.global_rela = torch.nn.Linear(emb_dim, self.med_voc_size)

        self.mha_sub_global = SAB(self.emb_dim, nheads=2, norm=True)
        self.mha_sub_sub = SAB(self.emb_dim, nheads=2, norm=True)
        self.mha_global_sub = SAB(self.emb_dim, nheads=2, norm=True)
        self.mha_global_global = SAB(self.emb_dim, nheads=2, norm=True)
        self.mha_final_context = SAB(self.emb_dim, nheads=2, norm=True)

        self.score_extractor = torch.nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)
        )

    def forward(
        self,
        substruct_data,
        mol_data,
        patient_data,
        ddi_mask_H,
        tensor_ddi_adj,
        average_projection,
    ):
        query, history_keys, history_values = self.patient_encoder(patient_data)
        # ==============> Prior Knowledge Graph
        prior_context = self.prior_encoder(query, history_keys, history_values)
        # <============== Prior Knowledge Graph

        # ==============> molecule context
        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)
        substruct_embeddings = self.substruct_encoder(**substruct_data).unsqueeze(0)

        mole_context = self.aggregate_mole_context(
            query.unsqueeze(0),
            global_embeddings.unsqueeze(0),
            substruct_embeddings,
            ddi_mask_H,
        ).squeeze(0)
        # <============== molecule context

        h = torch.cat([prior_context, mole_context], -1)

        logits = self.score_extractor(h).t()

        neg_pred_prob = torch.sigmoid(logits)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()

        return logits, batch_neg

    def aggregate_mole_context(self, query, global_emb, sub_emb, ddi_mask_H):
        global_weight = torch.sigmoid(self.global_rela(query.squeeze(0).squeeze(0)))
        global_weight = torch.diag(global_weight)

        weithted_global_emb = torch.matmul(global_weight, global_emb)

        substruct_weight = torch.sigmoid(
            self.substruct_rela(query.squeeze(0).squeeze(0))
        )
        substruct_weight = torch.diag(substruct_weight)

        weighted_sub_emb = torch.matmul(substruct_weight, sub_emb)

        global_emb_cross, _ = self.mha_global_sub(
            weithted_global_emb, weighted_sub_emb, ddi_mask_H
        )
        sub_emb_cross, _ = self.mha_sub_global(
            weighted_sub_emb, weithted_global_emb, ddi_mask_H.t()
        )

        global_emb_self, _ = self.mha_global_global(global_emb_cross, global_emb_cross)
        sub_emb_self, _ = self.mha_sub_sub(sub_emb_cross, sub_emb_cross)

        mole_context, global_local_weight = self.mha_final_context(
            global_emb_self, sub_emb_self, ddi_mask_H
        )

        return mole_context


class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)

    def forward(
        self, global_embeddings, substruct_embeddings, substruct_weight, mask=None
    ):
        Q = self.Qdense(global_embeddings)
        K = self.Kdense(substruct_embeddings)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = Attn.masked_fill(mask == 0, float("-inf"))

        Attn = torch.softmax(Attn, dim=-1)

        substruct_weight = torch.diag(substruct_weight)
        substruct_embeddings = torch.matmul(substruct_weight, substruct_embeddings)
        O = torch.matmul(Attn, substruct_embeddings)

        return O


class GCN(nn.Module):
    def __init__(
        self, voc_size, emb_dim, adj, dropout_rate=0.3, device=torch.device("cpu:0")
    ):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(
            h, self.W
        )  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
