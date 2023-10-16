import os
import torch
import random
import numpy as np
from collections import defaultdict
import json

class DataLoader:
    def __init__(self, params, saved_relation2id=None):
        self.task_dir = params.task_dir
        self.dataset = params.dataset

        ddi_paths = {
            'train': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'train')),
            'valid': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'valid')),
            'test':  os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'test')),
        }
        kg_paths = {
            'train': os.path.join(self.task_dir, 'data/{}/{}_KG.txt'.format(params.dataset, 'train')),
            'valid': os.path.join(self.task_dir, 'data/{}/{}_KG.txt'.format(params.dataset, 'valid')),
            'test':  os.path.join(self.task_dir, 'data/{}/{}_KG.txt'.format(params.dataset, 'test'))
        }
        
        self.process_files_ddi(ddi_paths, saved_relation2id)
        self.process_files_kg(kg_paths, saved_relation2id)
        self.load_ent_id()

        self.shuffle_train()
        fact_triplets = []
        for triplet in self.pos_triplets['train']:
            h, t, r = triplet[0], triplet[1], triplet[2:]
            for s in np.nonzero(r)[0]:
                fact_triplets.append([h,t,s])
        self.vKG = self.load_graph(np.array(fact_triplets), self.valid_kg)

        for triplet in self.pos_triplets['valid']:
            h, t, r = triplet[0], triplet[1], triplet[2:]
            for s in np.nonzero(r)[0]:
                fact_triplets.append([h,t,s])
        self.tKG = self.load_graph(np.array(fact_triplets), self.test_kg)

    def process_files_ddi(self, file_paths, saved_relation2id=None):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id

        self.pos_triplets = {}
        self.neg_triplets = {}
        self.train_ent = set()
        self.ent_pair = set()

        for file_type, file_path in file_paths.items():
            pos_triplet = []
            neg_triplet = []
            with open(file_path, 'r') as f:
                for line in f:
                    x, y, z, w = line.strip().split('\t')
                    x, y, w = int(x), int(y), int(w)
                    z1 = list(map(int, z.split(',')))
                    z = [i for i, _ in enumerate(z1) if _ == 1]
                    for s in z:
                        if x not in entity2id:
                            entity2id[x] = x
                        if y not in entity2id:
                            entity2id[y] = y
                        if not saved_relation2id and s not in relation2id:
                            relation2id[s] = s

                    if w==1:
                        pos_triplet.append([x,y] + z1)
                        self.ent_pair.add((x,y))
                        self.ent_pair.add((y,x))
                    else:
                        neg_triplet.append([x,y] + z1)
                
                    if file_type == 'train':
                        self.train_ent.add(x)
                        self.train_ent.add(y)
            self.pos_triplets[file_type] = np.array(pos_triplet, dtype='int')
            self.neg_triplets[file_type] = np.array(neg_triplet, dtype='int')

        self.entity2id = entity2id
        self.relation2id = relation2id

        self.eval_ent = max(self.entity2id.keys()) + 1
        self.eval_rel = len(self.relation2id)

    def load_ent_id(self, ):
        id2entity = dict()
        id2relation = dict()
        drug_set = json.load(open(os.path.join(self.task_dir, 'data/id2drug.json'), 'r'))
        entity_set = json.load(open(os.path.join(self.task_dir, 'data/entity2id.json'), 'r'))
        relation_set = json.load(open(os.path.join(self.task_dir, 'data/relation2id.json'), 'r'))
        for drug in drug_set:
            id2entity[int(drug)] = drug_set[drug]['cid']
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent

        for rel in relation_set:
            id2relation[int(rel)] = relation_set[rel]
        
        self.id2entity = id2entity
        self.id2relation = id2relation

    def process_files_kg(self, kg_paths, saved_relation2id=None, ratio=1):
        self.kg_triplets = defaultdict(list)
        self.ddi_in_kg = set()
        print('pruned ratio of edges in KG: {}'.format(ratio))

        for file_type, file_path in kg_paths.items():
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]

                for triplet in file_data:
                    h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if h not in self.entity2id:
                        self.entity2id[h] = h
                    if t not in self.entity2id:
                        self.entity2id[t] = t
                    if not saved_relation2id and r not in self.relation2id:
                        self.relation2id[r] = r
                    self.kg_triplets[file_type].append([h, t, r])
                    if h in self.train_ent:
                        self.ddi_in_kg.add(h)
                    if t in self.train_ent:
                        self.ddi_in_kg.add(t)

        if ratio < 1:
            n_train = len(self.kg_triplets['train'])
            n_valid = len(self.kg_triplets['valid'])
            n_test = len(self.kg_triplets['valid'])
            self.kg_triplets['train'] = random.sample(self.kg_triplets['train'], int(ratio*n_train))
            self.kg_triplets['valid'] = random.sample(self.kg_triplets['valid'], int(ratio*n_valid))
            self.kg_triplets['test'] = random.sample(self.kg_triplets['test'], int(ratio*n_test))

        train_kg = self.kg_triplets['train']
        valid_kg = train_kg + self.kg_triplets['valid']
        test_kg  = valid_kg + self.kg_triplets['test']
        self.train_kg = np.array(train_kg, dtype='int')
        self.valid_kg = np.array(valid_kg, dtype='int')
        self.test_kg = np.array(test_kg, dtype='int')
        print("KG triplets: Train-{} Valid-{} Test-{}".format(len(train_kg), len(valid_kg), len(test_kg)))

        self.all_ent = max(self.entity2id.keys()) + 1
        self.all_rel = max(self.relation2id.keys()) + 1

    def load_graph(self, triplets, kg_triplets=None):
        new_triplets = []
        for triplet in triplets:
            h, t, r = triplet
            new_triplets.append([t, h, 0])
            new_triplets.append([h, t, 0])
        if kg_triplets is not None:
            for triplet in kg_triplets:
                h, t, r = triplet
                r_inv = r + self.all_rel-self.eval_rel
                new_triplets.append([t, h, r])
                new_triplets.append([h, t, r_inv])
        edges = np.array(new_triplets)
        all_rel = 2*self.all_rel - self.eval_rel
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        n_ent = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(np.random.choice(list(self.ddi_in_kg), n_ent-int(n_ent*ratio)))
        all_triplet = np.array(self.pos_triplets['train'])
        if self.dataset.startswith('S1'):
            fact_triplet = []
            self.train_pos = []
            self.train_neg = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i,0], all_triplet[i,1], all_triplet[i,2:]
                if h in train_ent and t in train_ent:
                    for s in np.nonzero(r)[0]:
                        fact_triplet.append([h, t, s])
                elif h in train_ent or t in train_ent:
                    self.train_pos.append(self.pos_triplets['train'][i])
                    self.train_neg.append(self.neg_triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.train_pos = np.array(self.train_pos)
            self.train_neg = np.array(self.train_neg)
            self.KG = self.load_graph(fact_triplet, self.train_kg)
        elif self.dataset.startswith('S2'):
            fact_triplet = []
            self.train_pos = []
            self.train_neg = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i,0], all_triplet[i,1], all_triplet[i,2:]
                if h in train_ent and t in train_ent:
                    for s in np.nonzero(r)[0]:
                        fact_triplet.append([h, t, s])
                elif h not in train_ent and t not in train_ent:
                    self.train_pos.append(self.pos_triplets['train'][i])
                    self.train_neg.append(self.neg_triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.train_pos = np.array(self.train_pos)
            self.train_neg = np.array(self.train_neg)
            self.KG = self.load_graph(fact_triplet, self.train_kg)
        elif self.dataset.startswith('S0'):
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            n_fact = int(n_all * 0.8)

            facts = all_triplet[rand_idx[:n_fact]]
            self.train_pos = all_triplet[rand_idx[n_fact:]]
            self.train_neg = self.neg_triplets['train'][rand_idx[n_fact:]]
            fact_triplet = []

            for i in range(n_fact):
                x, y, z = facts[i,0], facts[i,1], facts[i,2:]
                for s in np.nonzero(z)[0]:
                    fact_triplet.append([x, y, s])
            fact_triplet = np.array(fact_triplet)
            self.KG = self.load_graph(fact_triplet, self.train_kg)

    def double_triple(self, triplet):
        new_triples = []
        n_rel = self.all_rel
        for triple in triplet:
            h, t, r = triple
            new_triples.append([t, h, r])
            new_triples.append([h, t, r+n_rel])
        new_triples = np.array(new_triples)
        return new_triples
