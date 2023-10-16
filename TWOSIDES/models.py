import torch
import torch.nn as nn
import numpy as np
from torchdrug.layers import functional
import pickle
from torch_scatter import scatter_add


class EmerGNN(nn.Module):
    def __init__(self, eval_ent, eval_rel, args):
        super(EmerGNN, self).__init__()
        self.eval_ent = eval_ent
        self.eval_rel = eval_rel
        self.all_ent = args.all_ent
        self.all_rel = args.all_rel
        self.L = args.length
        all_rel = 2*args.all_rel - args.eval_rel + 1
        with open('data/id2drug_feat.pkl', 'rb') as f:
            x = pickle.load(f, encoding='utf-8')
            mfeat = [] 
            for k in x:
                mfeat.append(x[k]['Morgan'])
            mfeat = np.array(mfeat)
        if args.feat == 'M':
            self.ent_kg = nn.Parameter(torch.FloatTensor(mfeat), requires_grad=False)
            self.Went = nn.Linear(1024, args.n_dim)
            self.Wr = nn.Linear(2*args.n_dim, eval_rel)
        else:
            self.ent_kg = nn.Embedding(eval_ent, args.n_dim)
            self.Wr = nn.Linear(4*args.n_dim, eval_rel)
        self.rel_kg = nn.ModuleList([nn.Embedding(all_rel, args.n_dim) for i in range(self.L)])

        self.args = args
        self.n_dim = args.n_dim
        self.linear = nn.ModuleList([nn.Linear(args.n_dim, args.n_dim) for i in range(self.L)])
        self.W = nn.Linear(args.n_dim, 1)
        self.init_weight()
        self.act = nn.ReLU()

        self.relation_linear = nn.ModuleList([nn.Linear(2*args.n_dim, 5) for i in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, all_rel) for i in range(self.L)])


    def init_weight(self):
        for param in self.parameters():
            if param.data.ndim>1 and param.requires_grad:
                nn.init.xavier_uniform_(param.data)

    def enc_ht(self, head, tail, KG):
        if self.args.feat == 'E':
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        else:
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])
        n_ent = self.all_ent
        
        # propagate from u to v
        hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
        hiddens[head, torch.arange(len(head)).cuda()] = head_embed
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight       # (1, n_rel, n_dim)
            relation_input = relation_weight * rel_embed    #(batch_size, n_rel, n_dim)
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0,1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))]

        # propagate from v to u
        hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
        hiddens[tail, torch.arange(len(tail)).cuda()] = tail_embed
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight       # (1, n_rel, n_dim)
            relation_input = relation_weight * rel_embed    #(batch_size, n_rel, n_dim)
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0,1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        head_hid = hiddens.view(n_ent, len(head), -1)[head, torch.arange(len(head))]

        if self.args.feat == 'E':
            embeddings = torch.cat([head_embed, head_hid, tail_hid, tail_embed], dim=1)
        else:
            embeddings = torch.cat([head_hid, tail_hid], dim=1)
        return embeddings

    def enc_r(self, ht_embed):
        scores = self.Wr(ht_embed)
        return scores

    def visualize_forward(self, head, tail, rela, KG, num_beam=5, head_batch=True):
        assert head.numel() == 1 and head.ndim == 1

        if self.args.feat == 'E':
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        else:
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])

        edge_index = KG._indices().t()

        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        step_weights = []
        edges_rel = KG._indices()[2]
        for l in range(self.L):
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight)
            edge_weights = relation_weight.squeeze(0)[edges_rel]
            step_weights.append(edge_weights)

        paths, weights = self.visualize(head, tail, edge_index, step_weights, num_beam=num_beam, head_batch=head_batch)
        
        return paths, weights

    def visualize(self, head, tail, edge_index, edge_weights, num_beam=5, head_batch=True):
        n_ent = self.all_ent
        inputs = torch.full((n_ent, num_beam), float('-inf')).cuda()
        if head_batch:
            inputs[head, 0] = 0
        else:
            inputs[tail, 0] = 0
        distances = []
        back_edges = []

        for i in range(len(edge_weights)):
            if head_batch:
                edge_mask = edge_index[:,0] != tail
            else:
                edge_mask = edge_index[:,0] != head
            edge_step = edge_index[edge_mask]
            node_in, node_out = edge_step.t()[:2]

            message = inputs[node_in] + edge_weights[i][edge_mask].unsqueeze(-1)  # [n_edge_step, num_beam]   this is the accumulated beam score
            msg_source = edge_step.unsqueeze(1).expand(-1, num_beam, -1)    # [n_edge_step, num_beam, 3]

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - torch.arange(num_beam, dtype=torch.float).cuda() / num_beam     # [n_edge, num_beam, num_beam]
            prev_rank = is_duplicate.argmax(dim=-1,keepdim=True)    # [n_edge, num_beam, 1]
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # [n_edge, num_bearm, 4]

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)   
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)

            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=n_ent)     # [n_ent]       # how many in-edges per node
            sizes = size[node_out_set] * num_beam
            arange = torch.arange(len(sizes)).cuda()
            msg2out = arange.repeat_interleave(sizes)
            #msg2out= functional._size_to_index(size[node_out_set] * num_beam)     

            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool).cuda(), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))
            #print(i, 'edges after remove dup', len(message))
            
            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=n_ent)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=n_ent)
            else:
                #print('second branch')
                distance = torch.full((n_ent, num_beam), float('-inf')).cuda()
                back_edge = torch.zeros(n_ent, num_beam, 4).long().cuda()
           
            distances.append(distance)
            back_edges.append(back_edge)
            inputs = distance

        # get topk_average_length
        k = num_beam
        paths = []
        weights = []

        for i in range(len(distances)):
            if head_batch:
                distance, order = distances[i][tail].flatten(0,-1).sort(descending=True)
                back_edge = back_edges[i][tail].flatten(0, -2)[order]
            else:
                distance, order = distances[i][head].flatten(0,-1).sort(descending=True)
                back_edge = back_edges[i][head].flatten(0, -2)[order]
            for d, (h,t,r,prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float('-inf'):
                    break
                path = [(h,t,r)]
                for j in range(i-1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                weights.append(d / len(path))

        if paths:
            weights, paths = zip(*sorted(zip(weights, paths), reverse=True)[:k])
        
        return paths, weights

    def get_attention_weights(self, head, tail, KG):
        if self.args.feat == 'E':
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        else:
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])
        
        outputs = []
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        for l in range(self.L):
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight)
            outputs.append(relation_weight.cpu().data.numpy())
        return outputs
