import numpy as np
from torch_geometric.utils import accuracy


def confidence_labels(data, labels, x_c, params):
    """
    a method of getting data of pseudo labels
    :param data:
    :param labels:
    :param x_c:
    :param params:
    :return: pmask: the data for pseudo labels of masks, plabels: pseudo labels
    """
    x_c_tc = []
    pmask, plabels = [], labels
    for cls in params['test_classes']:
        mask = x_c[params['masks'][cls], :]
        x_c_tc.append(mask)
    for i in range(params['n_way']):
        mask_tc = x_c_tc[i]
        mask_tc_sort = mask_tc[mask_tc[:, 1].argsort()]

        for j in range(len(mask_tc)):
            for k in params['test_mask_spt']:
                if k == mask_tc_sort[j][0]:
                    mask_tc_sort = np.delete(mask_tc_sort, j, axis=0)
                continue
            if j < params['self_train_num']:
                plabels[j] = mask_tc_sort[j][2]
                pmask.append(mask_tc_sort[j][0])
            else:
                break
    data.yy = plabels
    params['pmask'] = pmask

    return data, plabels, pmask


def get_acc(model, data, labels, meta_test_mask):
    pred = model(data).max(dim=1)[1]
    acc = accuracy(pred[meta_test_mask], labels[meta_test_mask])
    return acc
