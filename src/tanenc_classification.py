"""UniFi: train/evaluate the time-aware DNN on CommCSI-HAR.

Trains :class:`models.enc_mtan_classif_csi` over ``--seed-len`` random seeds and
reports, per seed and aggregated (mean +/- std), the test accuracy at
best-val-loss, best-val-acc, best-test-acc and at the last epoch.

The sensing configuration is fixed to the one reported in the paper -- two
synchronized nodes, all three spatial streams each, burst-filtered at 10 ms --
so the headline result needs no arguments at all::

    python tanenc_classification.py
"""

import argparse
import logging
import os
import time
from argparse import BooleanOptionalAction
from random import SystemRandom

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import models
import utils

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
# data
parser.add_argument('--data-root', type=str, default='./data',
                    help='folder holding the sanitized CommCSI-HAR splits')
parser.add_argument('--csi-subcarrier-num', type=int, default=208)
parser.add_argument('--subcarrier-motion-stats', action=BooleanOptionalAction,
                    default=True,
                    help='keep only motion-informative subcarriers (ISS)')
parser.add_argument('--normalize-input', action=BooleanOptionalAction,
                    default=True,
                    help='per-subcarrier z-score normalization')
# model
parser.add_argument('--rec-hidden', type=int, default=64, help='GRU hidden size')
parser.add_argument('--embed-time', type=int, default=64, help='time-embedding dim')
parser.add_argument('--query-points-num', type=int, default=64,
                    help='number of reference query points')
parser.add_argument('--num-heads', type=int, default=1)
parser.add_argument('--freq', type=float, default=100.,
                    help='base frequency of the fixed sinusoidal time embedding')
parser.add_argument('--learn-emb', action=BooleanOptionalAction, default=True,
                    help='learn the sinusoidal time embedding')
# training
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--niters', type=int, default=150)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--seed-len', type=int, default=5,
                    help='number of consecutive seeds to run')
# bookkeeping
parser.add_argument('--modelsave', type=int, default=0)
parser.add_argument('--logsave', type=int, default=0)
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print('\n', args, experiment_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.wandb:
        import wandb

    if args.modelsave or args.logsave:
        result_folder_path = 'result'
        result_file_prefix = (f'subc{args.csi_subcarrier_num}_'
                              f'mtan_enc_csi_{experiment_id}')
        os.makedirs(result_folder_path, exist_ok=True)
        with open(f'{result_folder_path}/{result_file_prefix}_args.txt', 'a') as f:
            print("{}".format(vars(args)), file=f)

    best_val_loss_all = []
    best_val_loss_test_acc_all = []
    best_val_loss_itr_all = []
    best_val_acc_all = []
    best_val_acc_test_acc_all = []
    best_val_acc_itr_all = []
    best_test_acc_all = []
    best_test_acc_val_loss_all = []
    best_test_acc_val_acc_all = []
    best_test_acc_itr_all = []
    early_stopping_test_acc_all = []
    early_stopping_test_acc_itr_all = []
    last_epoch_acc_all = []

    for randomseed in range(args.seed, args.seed + args.seed_len):
        print(f'\n\n------------ experiment_id: {experiment_id}, '
              f'randomseed: {randomseed} -------------\n')
        if args.wandb:
            wandb.init(project="unifi-commcsi-har",
                       name=f"seed{randomseed}",
                       config=args, notes=str(experiment_id))
        if args.logsave:
            with open(f'{result_folder_path}/{result_file_prefix}_log.txt', 'a') as f:
                print(f'\n\n------------ experiment_id: {experiment_id}, '
                      f'randomseed: {randomseed} -------------\n', file=f)

        torch.manual_seed(randomseed)
        np.random.seed(randomseed)
        torch.cuda.manual_seed(randomseed)

        data_obj = utils.get_testbed_irregular_ts_csi(args, 'cpu')
        train_loader = data_obj["train_dataloader"]
        val_loader = data_obj["val_dataloader"]
        test_loader = data_obj["test_dataloader"]
        dim = data_obj["input_dim"]

        print(f"\ninputs dim: {dim}")
        print(f"outputs dim: {data_obj['n_labels']}")

        # torch.linspace(0, 1., N) holds the reference/query time points
        rec = models.enc_mtan_classif_csi(
            dim, torch.linspace(0, 1., args.query_points_num), args.rec_hidden,
            data_obj["n_labels"], args.embed_time, args.num_heads, args.learn_emb,
            args.freq, device).to(device)

        print('\nparameters:', utils.count_parameters(rec), '\n')
        # Constant learning rate: the published runs built a CosineAnnealingLR
        # but never stepped it, so no scheduler is used here.
        optimizer = optim.AdamW(list(rec.parameters()), lr=args.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val_loss, best_val_loss_test_acc = float('inf'), float('-inf')
        best_val_acc, best_val_acc_test_acc = float('-inf'), float('-inf')
        best_test_acc, best_test_acc_val_loss, best_test_acc_val_acc = \
            float('-inf'), float('inf'), float('-inf')
        best_val_loss_itr = best_val_acc_itr = best_test_acc_itr = 0
        last_epoch_acc = float('-inf')
        early_stopping_test_acc = 0
        early_stopping_test_acc_itr = 0
        patience = 0
        total_time = 0.

        for itr in tqdm(range(1, args.niters + 1)):
            train_loss, train_acc, train_n = 0, 0, 0
            start_time = time.time()
            for train_batch, label in train_loader:
                train_batch, label = train_batch.to(device), label.to(device)
                batch_len = train_batch.shape[0]
                observed_data, observed_mask = \
                    train_batch[:, :, :dim], train_batch[:, :, dim:2*dim]
                observed_tp = train_batch[:, :, -1]
                out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_len
                train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
                train_n += batch_len
            total_time += time.time() - start_time

            val_loss, val_acc, val_auc, val_cm = utils.evaluate_classifier(
                rec, val_loader, dim=dim, device=device)
            test_loss, test_acc, test_auc, test_cm = utils.evaluate_classifier(
                rec, test_loader, dim=dim, device=device)

            if args.logsave == 0:
                tqdm.write(f'Iter: {itr}, loss: {train_loss/train_n:.4f}, '
                           f'acc: {train_acc/train_n:.4f}, val_loss: {val_loss:.4f}, '
                           f'val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}, '
                           f'test_auc: {test_auc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_test_acc = test_acc
                best_val_loss_itr = itr
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc_test_acc = test_acc
                best_val_acc_itr = itr
                patience = 0
            else:
                patience += 1
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_val_loss = val_loss
                best_test_acc_val_acc = val_acc
                best_test_acc_itr = itr
            last_epoch_acc = test_acc

            if args.wandb:
                wandb.log({"train_loss": train_loss/train_n,
                           "train_acc": train_acc/train_n, "val_loss": val_loss,
                           "val_acc": val_acc, "test_acc": test_acc,
                           "test_auc": test_auc})
            if args.logsave:
                with open(f'{result_folder_path}/{result_file_prefix}_log.txt', 'a') as f:
                    print('Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, '
                          'val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
                          .format(itr, train_loss/train_n, train_acc/train_n,
                                  val_loss, val_acc, test_acc, test_auc), file=f)
            if args.modelsave and itr % 100 == 0:
                ckpt = f'{result_folder_path}/{result_file_prefix}.h5'
                torch.save({'args': args, 'epoch': itr,
                            'rec_state_dict': rec.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': -loss}, ckpt)
                print('Model saved to: ', ckpt)

            # note: recorded, not enforced -- training runs the full --niters
            if patience > 20 and early_stopping_test_acc == 0:
                early_stopping_test_acc = best_val_acc_test_acc
                early_stopping_test_acc_itr = itr
                print('Early stopping point, best_val_acc: ', best_val_acc,
                      'best_val_acc_test_acc: ', best_val_acc_test_acc)

        print('\nbest_val_loss: ', best_val_loss, 'best_val_loss_test_acc: ',
              best_val_loss_test_acc, 'best_val_loss_itr: ', best_val_loss_itr)
        print('best_val_acc:  ', best_val_acc, 'best_val_acc_test_acc:  ',
              best_val_acc_test_acc, 'best_val_acc_itr: ', best_val_acc_itr)
        print('best_test_acc: ', best_test_acc, 'best_test_acc_val_loss: ',
              best_test_acc_val_loss, 'best_test_acc_val_acc: ',
              best_test_acc_val_acc, 'best_test_acc_itr: ', best_test_acc_itr)
        print('early_stopping_test_acc: ', early_stopping_test_acc,
              'early_stopping_test_acc_itr: ', early_stopping_test_acc_itr)
        print('last_epoch_acc: ', last_epoch_acc)
        print('last val_cm: \n', val_cm)
        print('last test_cm: \n', test_cm)
        print('\nexperiment_id: ', experiment_id, ', randomseed: ', randomseed,
              ', total_time: ', total_time, 'seconds\n\n')

        if args.wandb:
            wandb.log({"best_val_loss": best_val_loss,
                       "best_val_loss_test_acc": best_val_loss_test_acc,
                       "best_val_acc": best_val_acc,
                       "best_val_acc_test_acc": best_val_acc_test_acc,
                       "best_test_acc": best_test_acc,
                       "last_epoch_acc": last_epoch_acc})
            wandb.finish()
        if args.logsave:
            with open(f'{result_folder_path}/{result_file_prefix}_log.txt', 'a') as f:
                print('\nbest_val_loss: ', best_val_loss, 'best_val_loss_test_acc: ',
                      best_val_loss_test_acc, file=f)
                print('best_val_acc: ', best_val_acc, 'best_val_acc_test_acc: ',
                      best_val_acc_test_acc, file=f)
                print('best_test_acc: ', best_test_acc, file=f)
                print('last_epoch_acc: ', last_epoch_acc, file=f)
                print('total_time: ', total_time, 'seconds\n\n', file=f)

        best_val_loss_all.append(best_val_loss)
        best_val_loss_test_acc_all.append(best_val_loss_test_acc)
        best_val_loss_itr_all.append(best_val_loss_itr)
        best_val_acc_all.append(best_val_acc)
        best_val_acc_test_acc_all.append(best_val_acc_test_acc)
        best_val_acc_itr_all.append(best_val_acc_itr)
        best_test_acc_all.append(best_test_acc)
        best_test_acc_val_loss_all.append(best_test_acc_val_loss)
        best_test_acc_val_acc_all.append(best_test_acc_val_acc)
        best_test_acc_itr_all.append(best_test_acc_itr)
        early_stopping_test_acc_all.append(early_stopping_test_acc)
        early_stopping_test_acc_itr_all.append(early_stopping_test_acc_itr)
        last_epoch_acc_all.append(last_epoch_acc)

    print('\nbest_val_loss_all:', [f'{v:.4f}' for v in best_val_loss_all])
    print('best_val_loss_test_acc_all:', [f'{v:.4f}' for v in best_val_loss_test_acc_all])
    print('best_val_loss_itr_all:', best_val_loss_itr_all)
    print('best_val_acc_all:', [f'{v:.4f}' for v in best_val_acc_all])
    print('best_val_acc_test_acc_all:', [f'{v:.4f}' for v in best_val_acc_test_acc_all])
    print('best_val_acc_itr_all:', best_val_acc_itr_all)
    print('best_test_acc_all:', [f'{v:.4f}' for v in best_test_acc_all])
    print('best_test_acc_itr_all:', best_test_acc_itr_all)
    print('early_stopping_test_acc_all:', [f'{v:.4f}' for v in early_stopping_test_acc_all])
    print('last_epoch_acc_all:', [f'{v:.4f}' for v in last_epoch_acc_all])

    for name, values in [
            ('best_val_loss_test_acc_all_array', best_val_loss_test_acc_all),
            ('best_val_acc_test_acc_all_array ', best_val_acc_test_acc_all),
            ('best_test_acc_all_array         ', best_test_acc_all),
            ('early_stopping_test_acc_all_array', early_stopping_test_acc_all),
            ('last_epoch_acc_all_array        ', last_epoch_acc_all)]:
        a = np.array(values)
        print("{} - mean:{:.4f}, std:{:.4f}".format(name, a.mean(), a.std()))

    print('\nparameters:', utils.count_parameters(rec))
    print('\n', args, experiment_id)

    if args.logsave:
        with open(f'{result_folder_path}/{result_file_prefix}_stats.txt', 'a') as f:
            print('best_val_loss_test_acc_all:',
                  [f'{v:.4f}' for v in best_val_loss_test_acc_all], file=f)
            print('best_val_acc_test_acc_all:',
                  [f'{v:.4f}' for v in best_val_acc_test_acc_all], file=f)
            print('best_test_acc_all:', [f'{v:.4f}' for v in best_test_acc_all], file=f)
            print('last_epoch_acc_all:', [f'{v:.4f}' for v in last_epoch_acc_all], file=f)
