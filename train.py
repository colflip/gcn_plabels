import torch.nn.functional as F


def gcn_train(model, optimizer, data, labels, mask_train, params):
    model.train()
    min_loss = 9999
    patience = 0
    for epoch in range(params['epoch']):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[mask_train], labels[mask_train])
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            min_loss = loss
            patience = 0
        else:
            patience = patience + 1
        if patience > params['patience']:
            break
    weights = [model.conv1.lin.weight, model.conv2.lin.weight]
    return weights


def gcn_plabels_train(model, optimizer, data, labels, mask_train, params):
    model.train()
    min_loss = 9999
    patience = 0
    for epoch in range(params['epoch']):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[mask_train], labels[mask_train])
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            min_loss = loss
            patience = 0
        else:
            patience = patience + 1
        if patience > params['patience']:
            break
    return
