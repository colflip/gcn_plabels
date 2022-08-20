import random

from torch_geometric.datasets import Coauthor, Amazon, Planetoid, CoraFull


def load_data(params):
    d_name = params['dataset']
    if d_name == "Cora":
        dataset = Planetoid(name='Cora', root='./data')
    elif d_name == 'CiteSeer':
        dataset = Planetoid(name='Citeseer', root='./data')
    elif d_name == 'cs':
        dataset = Coauthor(name='cs', root='./data')
    elif d_name == 'Computers':
        dataset = Amazon(name='Computers', root='./data')
    elif d_name == 'Photo':
        dataset = Amazon(name='Photo', root='./data')
    elif d_name == 'CoraFull':
        dataset = CoraFull(root='./data/CoraFull')
    else:
        return None
    data = dataset[0].to(params['device'])
    labels = data.y.clone().detach()
    num_classes = dataset.num_classes
    node_num = len(data.y)

    params['n_way'] = num_classes * 2 // 5
    all_classes = [i for i in range(num_classes)]
    test_classes = random.sample(all_classes, params['n_way'])
    params['all_classes'] = all_classes
    params['test_classes'] = test_classes
    train_classes = [i for i in all_classes if i not in test_classes]
    params['train_classes'] = train_classes

    params['in_dim'] = dataset.num_features
    params['out_dim'] = num_classes

    masks = [[] for i in range(num_classes)]
    for i in range(node_num):
        cls = labels[i]
        masks[cls].append(i)

    meta_train_mask = []
    for cls in train_classes:
        meta_train_mask.extend(masks[cls])

    meta_test_mask_spt = []
    meta_test_mask_qry = []
    for cls in test_classes:
        spt = random.sample(masks[cls], params['spt_num'])
        meta_test_mask_spt.extend(spt)
        if len(masks[cls]) > params['qry_num']:
            qry = random.sample(masks[cls], params['qry_num'])
        else:
            qry = masks[cls].copy()
        meta_test_mask_qry.extend(qry)

    masks = [[] for i in range(num_classes)]
    for i in range(node_num):
        cls = labels[i]
        masks[cls].append(i)
    params['masks'] = masks
    params['train_mask'] = meta_train_mask
    params['test_mask_spt'] = meta_test_mask_spt
    params['test_mask_qry'] = meta_test_mask_qry
    return data, labels, meta_train_mask, meta_test_mask_spt, meta_test_mask_qry
