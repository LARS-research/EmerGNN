import os
import argparse
import torch
import random
from load_data import DataLoader

from base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser for EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
parser.add_argument('--dataset', type=str, default='emerging', help='the directory to dataset')
parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to load.')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size')
parser.add_argument('--seed', type=str, default=1234)
class options:
    def __init__():
        pass

if __name__ == '__main__':
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, eval_rel
    KG = dataloader.KG
    vKG = dataloader.vKG
    tKG = dataloader.tKG

    pos_triplets, neg_triplets = dataloader.pos_triplets, dataloader.neg_triplets
    train_pos, train_neg = torch.LongTensor(pos_triplets['train']).cuda(), torch.LongTensor(neg_triplets['train']).cuda()
    valid_pos, valid_neg = torch.LongTensor(pos_triplets['valid']).cuda(), torch.LongTensor(neg_triplets['valid']).cuda()
    test_pos,  test_neg  = torch.LongTensor(pos_triplets['test']).cuda(),  torch.LongTensor(neg_triplets['test']).cuda()

    if not os.path.exists('results'):
        os.makedirs('results')

    def run_model(params):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.lr = params['lr']
        args.lamb = params['lamb']
        args.length = params['length']
        args.n_batch = params['n_batch']
        args.n_dim = params['n_dim']
        args.feat = params['feat']

        model = BaseModel(eval_ent, eval_rel, args)
        best_acc = -1
        early_stop = 0
        try:
            for e in range(args.n_epoch):
                if early_stop > 3:
                    break
                dataloader.shuffle_train()
                KG = dataloader.KG
                train_pos, train_neg = dataloader.train_pos, dataloader.train_neg
                model.train(train_pos, train_neg, KG)
                if (e+1) % args.epoch_per_test == 0:
                    v_roc, v_pr, v_ap = model.evaluate(valid_pos, valid_neg, vKG)
                    t_roc, t_pr, t_ap = model.evaluate(test_pos, test_neg, tKG)
                    out_str = 'epoch:%d\tfeat:%s lr:%.6f lamb:%.8f n_batch:%d n_dim:%d layer:%d\t%.4f %.4f %.4f\t %.4f %.4f %.4f' % (e+1, args.feat, args.lr, args.lamb, args.n_batch, args.n_dim, args.length, v_roc, v_pr, v_ap, t_roc, t_pr, t_ap)
                    if v_pr > best_acc:
                        best_acc = v_pr
                        best_str = out_str
                        early_stop = 0
                    else:
                        early_stop += 1
        except RuntimeError as e:
            print(e)
            return 0
                    #print(out_str)

        print(best_str)
        with open(os.path.join('results', args.model+'_'+ args.dataset+'_tune.txt'), 'a+') as f:
            f.write(best_str+'\n\n')
        return -best_acc

    space = {
        "lr": hp.choice("lr", [1e-2, 3e-3, 1e-3, 3e-4]),
        "lamb": hp.choice("lamb", [1e-8, 1e-6, 1e-4, 1e-2]),
        "n_batch": hp.choice("n_batch", [32, 64, 128]),
        "n_dim": hp.choice("n_dim", [32, 64]),
        "length": hp.choice("length", [2,3,4,5]),
        "feat": hp.choice("feat", ['M', 'E']),
    }

    trials = Trials()
    best = fmin(run_model, space, algo=partial(tpe.suggest, n_startup_jobs=60), max_evals=100, trials=trials)
    print(best)
                

    

