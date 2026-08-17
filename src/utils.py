"""Data loading, batching and evaluation helpers for UniFi."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from testbed_dataset import Testbed_Dataset_IrregularNSS

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_classifier(model, test_loader, dim=41, device='cuda'):
    """Run the classifier over a loader and return (loss, accuracy, macro AUC,
    confusion matrix)."""
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    predicted_labels = pred.argmax(1)
    acc = np.mean(predicted_labels == true)
    cm = confusion_matrix(true, predicted_labels)

    n_classes = pred.shape[1]
    y_score = torch.softmax(torch.from_numpy(pred), dim=1).numpy()
    if n_classes == 2:
        # Binary: only compute if both classes present
        auc = metrics.roc_auc_score(true, y_score[:, 1]) \
            if np.unique(true).size == 2 else 0.0
    else:
        # Multi-class: macro OvR but skip classes missing in y_true
        y_true_one_hot = label_binarize(true, classes=np.arange(n_classes))
        aucs = []
        for c in range(n_classes):
            y_true_c = y_true_one_hot[:, c]
            if np.unique(y_true_c).size < 2:
                continue
            aucs.append(metrics.roc_auc_score(y_true_c, y_score[:, c]))
        auc = float(np.mean(aucs)) if len(aucs) else 0.0
    return test_loss/pred.shape[0], acc, auc, cm


def _time_series_stats(data, mask):
    channels, eps = data.shape[-1], 1e-7
    print("data.shape: ", data.shape)
    print("mask.shape: ", mask.shape)
    data = data.transpose((2, 0, 1)).reshape(channels, -1)
    # data = data.permute(2, 0, 1).reshape(channels, -1)

    mask = mask.transpose((2, 0, 1)).reshape(channels, -1)
    # mask = mask.permute(2, 0, 1).reshape(channels, -1)
    mu, std = np.zeros((channels, 1)), np.zeros((channels, 1))
    for c in range(channels):
        val = data[c, :][mask[c, :] == 1]
        mu[c], std[c] = np.mean(val), np.std(val)
        # mu[c], std[c] = val.mean().item(), val.std().item()
        std[c] = np.max([std[c][0], eps])
    return mu, std


def _time_normalize_raindrop(data, mask, time, mu, std, t_max):
    num, t, channels = data.shape
    data = data.transpose((2, 0, 1)).reshape(channels, -1)
    mask = mask.transpose((2, 0, 1)).reshape(channels, -1)
    for c in range(channels):
        data[c] = (data[c] - mu[c]) / (std[c] + 1e-18)
    data = data * mask
    data = data.reshape(channels, num, t).transpose((1, 2, 0))
    mask = mask.reshape(channels, num, t).transpose((1, 2, 0))
    time /= t_max
    return np.concatenate([data, mask, time], axis=-1)


def get_testbed_irregular_ts_csi(args, device):
    """Build train/val/test dataloaders for CommCSI-HAR.

    Each returned batch item is a single tensor holding
    ``[values | mask | timestamp]`` concatenated along the feature axis, which
    is what :class:`models.enc_mtan_classif_csi` expects.
    """
    # The classifier head keeps 7 logits although CommCSI-HAR has 6 activities;
    # this matches the published models (the 7th logit is never used).
    num_classes = 7

    split_dir = (f'{args.data_root}/csis_l2norm_normal_validsubcarriers_alignedNss01_Nss2'
                 f'_syncedNodes_segmented_stichNodesNSSs_splitDatasets/node12')

    ds_kwargs = dict(subcarrier_num=args.csi_subcarrier_num,
                     subcarrier_motion_stats=args.subcarrier_motion_stats)
    train_val_ds = Testbed_Dataset_IrregularNSS(split_dir + '/trainval', **ds_kwargs)
    test_ds = Testbed_Dataset_IrregularNSS(split_dir + '/test', **ds_kwargs)

    train_dataset, val_dataset = model_selection.train_test_split(
        train_val_ds, train_size=0.8, random_state=42, shuffle=True)

    train_data_combined = testbed_variable_time_collate_fn(train_dataset, device, classify=True)
    val_data_combined = testbed_variable_time_collate_fn(val_dataset, device, classify=True)
    test_data_combined = testbed_variable_time_collate_fn(test_ds, device, classify=True)

    print("\nTraining size:", train_data_combined[0].size(),
          "label:", train_data_combined[1].size())
    print("Validation size:", val_data_combined[0].size(),
          "label:", val_data_combined[1].size())
    print("Test size:", test_data_combined[0].size(),
          "label:", test_data_combined[1].size())
    input_dim = int((train_data_combined[0].size()[-1] - 1) / 2)

    if args.normalize_input:
        # Per-subcarrier z-score over observed samples only (Sec. IV-B).  Runs
        # on CPU over the whole tensor, so it can take a while before the first
        # epoch starts -- this is expected, not a hang.
        channels = input_dim
        train_np = train_data_combined[0].cpu().numpy()
        val_np = val_data_combined[0].cpu().numpy()
        test_np = test_data_combined[0].cpu().numpy()

        mu, std = _time_series_stats(train_np[:, :, :channels],
                                     train_np[:, :, channels:-1])
        t_max = np.max(train_np[:, :, -1])
        train_np = _time_normalize_raindrop(train_np[:, :, :channels], train_np[:, :, channels:-1], train_np[:, :, -1:], mu, std, t_max)
        val_np = _time_normalize_raindrop(val_np[:, :, :channels], val_np[:, :, channels:-1], val_np[:, :, -1:], mu, std, t_max)
        test_np = _time_normalize_raindrop(test_np[:, :, :channels], test_np[:, :, channels:-1], test_np[:, :, -1:], mu, std, t_max)

        train_data_combined = (torch.from_numpy(train_np), train_data_combined[1])
        val_data_combined = (torch.from_numpy(val_np), val_data_combined[1])
        test_data_combined = (torch.from_numpy(test_np), test_data_combined[1])

    train_data_combined = TensorDataset(train_data_combined[0], train_data_combined[1].long().squeeze())
    val_data_combined = TensorDataset(val_data_combined[0], val_data_combined[1].long().squeeze())
    test_data_combined = TensorDataset(test_data_combined[0], test_data_combined[1].long().squeeze())

    train_dataloader = DataLoader(train_data_combined, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data_combined, batch_size=args.batch_size, shuffle=False)

    return {"train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader),
            "n_labels": num_classes}


# Burst removal threshold (Sec. IV-B), in seconds.
BURSTY_INTERVAL = 0.01


def _filter_bursty(tt, vals, min_interval=BURSTY_INTERVAL):
    """Burst removal (Sec. IV-B): drop packets that arrive less than
    ``min_interval`` seconds after the last *kept* packet.  Bursts of
    back-to-back frames carry almost no new channel information but dominate
    the attention over a segment."""
    if tt.numel() <= 1:
        return tt, vals
    kept_indices = [0]
    last_kept_time = tt[0]
    for i in range(1, tt.shape[0]):
        if tt[i] - last_kept_time >= min_interval:
            kept_indices.append(i)
            last_kept_time = tt[i]
    return tt[kept_indices], vals[kept_indices]


def testbed_variable_time_collate_fn(batch, device=torch.device("cpu"),
                                     classify=False):
    """Burst-filter each stream, then pad a batch onto one union timeline.

    ``batch[i]`` is ``(ts_dict, csi_dict, label)`` with six streams: two
    synchronized nodes x three spatial streams. The per-stream timelines are
    merged into one sorted union timeline per segment; every stream writes its
    CSI into its own block of the feature axis and marks the corresponding mask
    entries, so the mask records which stream was actually heard at each
    instant.

    Node 1's NSS0 shares its subcarriers with the first columns of NSS1 (NSS1 is
    the wide band whose first 52 subcarriers overlap NSS0), so those two
    deliberately write into overlapping blocks; same for node 2.

    Returns ``[values | mask | timestamp]`` concatenated on the feature axis,
    plus the labels when ``classify``.
    """
    N = 1  # number of labels

    csi_dict_sample = batch[0][1]
    node1_D_nss0 = csi_dict_sample[0].shape[1]
    node1_D_nss1 = csi_dict_sample[1].shape[1]
    node1_D_nss2 = csi_dict_sample[2].shape[1]
    node2_D_nss0 = csi_dict_sample[3].shape[1]
    node2_D_nss1 = csi_dict_sample[4].shape[1]
    node2_D_nss2 = csi_dict_sample[5].shape[1]
    D = node1_D_nss1 + node1_D_nss2 + node2_D_nss1 + node2_D_nss2

    processed_batch = []
    all_union_lengths = []
    for (ts_dict, csi_dict, label) in batch:
        filtered_ts, filtered_csi = {}, {}
        for nss_idx in range(6):
            tt = torch.from_numpy(ts_dict[nss_idx]).float()
            vals = torch.from_numpy(csi_dict[nss_idx]).float()
            filtered_ts[nss_idx], filtered_csi[nss_idx] = _filter_bursty(tt, vals)
        union_ts = torch.unique(torch.cat([filtered_ts[i] for i in range(6)]),
                                sorted=True)
        all_union_lengths.append(union_ts.shape[0])
        processed_batch.append((filtered_ts, filtered_csi, label))

    maxlen = np.max(all_union_lengths) if all_union_lengths else 0
    batch = processed_batch

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (ts_dict, csi_dict, label) in enumerate(batch):
        union_ts = torch.unique(torch.cat([ts_dict[i] for i in range(6)]),
                                sorted=True).to(device)
        currlen = union_ts.shape[0]
        enc_combined_tt[b, :currlen] = union_ts

        # searchsorted maps each stream's timestamps onto the union timeline
        idx = [torch.searchsorted(union_ts, ts_dict[i].to(device)) for i in range(6)]

        # Node 1: NSS0 overlaps NSS1's first columns, then NSS2
        enc_combined_vals[b, idx[0], 0:node1_D_nss0] = csi_dict[0].to(device)
        enc_combined_mask[b, idx[0], 0:node1_D_nss0] = 1
        enc_combined_vals[b, idx[1], 0:node1_D_nss1] = csi_dict[1].to(device)
        enc_combined_mask[b, idx[1], 0:node1_D_nss1] = 1
        start = node1_D_nss1
        end = start + node1_D_nss2
        enc_combined_vals[b, idx[2], start:end] = csi_dict[2].to(device)
        enc_combined_mask[b, idx[2], start:end] = 1

        # Node 2: same layout, appended after node 1's blocks
        start = end
        enc_combined_vals[b, idx[3], start:start + node2_D_nss0] = csi_dict[3].to(device)
        enc_combined_mask[b, idx[3], start:start + node2_D_nss0] = 1
        enc_combined_vals[b, idx[4], start:start + node2_D_nss1] = csi_dict[4].to(device)
        enc_combined_mask[b, idx[4], start:start + node2_D_nss1] = 1
        start = start + node2_D_nss1
        enc_combined_vals[b, idx[5], start:start + node2_D_nss2] = csi_dict[5].to(device)
        enc_combined_mask[b, idx[5], start:start + node2_D_nss2] = 1

        if classify:
            combined_labels[b] = torch.tensor(label, dtype=torch.long).to(device)

    if torch.max(enc_combined_tt) != 0.:
        # segments are 4 s long -> scale timestamps into [0, 1]
        enc_combined_tt = enc_combined_tt / 4.0

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)

    if classify:
        return combined_data, combined_labels
    return combined_data
