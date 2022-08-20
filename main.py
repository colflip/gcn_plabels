import random

import numpy as np
import torch
from argument import args_parse, setup_seed
from dataloader import load_data
from models import GCN
from train import gcn_train
from utils import confidence_labels, get_acc

acc_list = []
loop = 100
for i in range(loop):
    print("------loop: {}-------".format(i))
    setup_seed(65782134 + i)
    params = args_parse()
    data, labels, meta_train_mask, meta_test_mask_spt, meta_test_mask_qry = load_data(params)

    model = GCN(params['in_dim'], params['hid_dim'], params['mid_dim'], params['out_dim']).to(params['device'])
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.classifier.parameters()}], lr=params['l_rate'], weight_decay=params['w_decay'])
    _ = gcn_train(model, optimizer, data, labels, meta_test_mask_spt, params)
    x_c = model.confidence(data)
    print("gcn2: {}".format(get_acc(model, data, labels, meta_test_mask_qry)))

    data, plabels, pmask = confidence_labels(data, labels, x_c, params)
    _ = gcn_train(model, optimizer, data, plabels, meta_test_mask_spt + pmask, params)
    acc = get_acc(model, data, labels, meta_test_mask_qry)
    print("plabels: {}".format(acc))

    acc_list.append(acc)
print("k-shot: {} n-way: {} {} {};".format(params['spt_num'], params['n_way'], params['l_rate'], params['w_decay']),
      end="")
print("acc mean/var: {}±{}".format(round(np.mean(acc_list) * 100, 2), round(np.var(acc_list) * 100, 2)))
