from __future__ import annotations

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def run_train(model, loader_train, loader_valid, batch_loss, options):
    def validate(loader: DataLoader):
        model.eval()
        total_vloss = 0
        with torch.no_grad():
            for i, (u, y) in enumerate(loader):
                u = u.to(options['device'])
                y = y.to(options['device'])
                out = model(u)
                vLoss_ = batch_loss(out, y)
                total_vloss += vLoss_.item()
        return total_vloss / len(loader)

    def train(optimizer):
        model.train()
        total_loss = 0
        for idx, (u, y) in enumerate(loader_train):
            u = u.to(options['device'])
            y = y.to(options['device'])
            optimizer.zero_grad()
            out = model(u)
            loss_ = batch_loss(out, y)
            loss_.backward()
            optimizer.step()
            total_loss += loss_.item()
        return total_loss / len(loader_train)

    if not os.path.exists(options['model_path']):
        os.makedirs(options['model_path'])

    best_vLoss = validate(loader_valid)
    all_tlosses = []
    all_vlosses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=options['init_lr'])
    if options['change_lr']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               min_lr=options['min_lr'],
                                                               factor=options['lr_factor'],
                                                               patience=options['lr_patience'] if 'lr_patience' in options else 10,
                                                               verbose=True)

    start_time = time.time()

    for epoch in range(0, options['n_epochs']):
        train_loss = train(optimizer)
        vLoss = validate(loader_valid)

        all_tlosses.append(train_loss)
        all_vlosses.append(vLoss)

        if options['change_lr']:
            scheduler.step(vLoss)

        if vLoss < best_vLoss:
            best_vLoss = vLoss
            best_epoch = epoch
            torch.save(model.state_dict(), options['model_path']+options['model_file_name'])
            print('Model has been saved! Epoch is {:3d}. Best loss is {:.6f}.'.
                  format(best_epoch, best_vLoss))

        # 打印信息
        print('Train Epoch: [{:3d}/{:3d}], \tLoss: {:.6f}''\tVal Loss: {:.6f}'.
              format(epoch+1, options['n_epochs'], train_loss, vLoss), flush=True)

    time_el = time.time() - start_time
    print('\nTotal learning time: {:2.0f}:{:2.0f} [min:sec]'.format(time_el // 60, time_el - 60 * (time_el // 60)))

    # plot_metrics loss
    index = np.arange(options['n_epochs'])
    all_tlosses = np.array(all_tlosses)
    all_vlosses = np.array(all_vlosses)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(index, all_tlosses)
    ax2.plot(index, all_vlosses)
    plt.savefig(options['model_path'] + options['loss_fig_name'])
    plt.close()
    # plt.show()



