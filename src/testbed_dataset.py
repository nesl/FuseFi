"""CommCSI-HAR dataset reader (irregularly sampled CSI, per spatial stream).

Each ``.mat`` file is one 4-second activity segment produced by the offline
MATLAB sanitization pipeline.  A file stores, per spatial stream (NSS), the CSI
amplitudes and the packet arrival timestamps -- the timestamps are *not*
uniformly spaced, which is the whole point of UniFi.

Files hold the synchronized node pair reported in the paper, with keys
``Node{1,2}_csis_NSS{0,1,2}`` and ``Node{1,2}_timestamps_NSS{0,1,2}``.
"""

import glob

import scipy.io as sio
import torch
from torch.utils.data import Dataset

# Activity folders, in label order: walking=0, running=1, ... circlingarm=5.
VALID_ACTIVITIES = ['walking', 'running', 'boxing',
                    'wavinghands', 'sitting', 'circlingarm']

# Motion-informative subcarriers (ISS, Sec. IV-B): indices selected offline from
# per-subcarrier motion statistics.  Keyed by (node, nss).
MOTION_STAT_SUBCARRIERS = {
    (1, 0): [3, 11, 14, 23, 24, 32, 38, 42, 51],
    (1, 1): [3, 11, 14, 23, 24, 32, 38, 42, 51, 57, 64, 66, 77, 78, 89, 93, 96,
             107, 108, 115, 124, 126, 132, 140, 145, 155, 161, 162, 169, 177,
             183, 191, 192, 199, 207],
    (1, 2): [0, 6, 14, 22, 29, 30, 37, 45, 53, 58, 61, 66, 74, 82, 89, 90, 98,
             106, 108, 114, 123, 127, 137, 143, 144, 152, 161, 162, 168, 176,
             184, 191, 192, 199, 207],
    (2, 0): [5, 6, 17, 18, 27, 30, 41, 45, 48],
    (2, 1): [5, 6, 17, 18, 27, 30, 41, 45, 48],
    (2, 2): [1, 9, 17, 18, 29, 34, 36, 42, 49],
}

# The six streams of the synchronized pair, in feature-axis order.
STREAMS = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def _select_subcarriers(csis, node, nss, subcarrier_num, motion_stats):
    """Truncate to ``subcarrier_num`` columns, then optionally keep only the
    motion-informative subcarriers (ISS)."""
    csis = csis[:, :subcarrier_num]
    if motion_stats:
        idx = MOTION_STAT_SUBCARRIERS.get((node, nss))
        if idx is None:
            raise ValueError(
                f'No motion-statistics subcarrier set for node {node}, nss {nss}')
        idx = [i for i in idx if i <= csis.shape[1] - 1]
        csis = csis[:, idx]
    return csis


class Testbed_Dataset_IrregularNSS(Dataset):
    """CommCSI-HAR segments with their native, irregular timestamps.

    Returns ``(timestamps, csis, label)``, where ``timestamps`` and ``csis`` are
    dicts keyed by stream index (0..5, see ``STREAMS``); the collate function
    builds the union timeline across streams.
    """

    def __init__(self, root_dir, subcarrier_num=52,
                 subcarrier_motion_stats=False):
        self.root_dir = root_dir
        self.subcarrier_num = subcarrier_num
        self.subcarrier_motion_stats = subcarrier_motion_stats

        # Every .mat under an activity folder is used and filenames are never
        # parsed, so segments can be named however the release names them.
        # Sorting keeps the dataset order -- and hence the train/val split drawn
        # from it -- independent of the order the filesystem hands files back.
        self.data_list = [
            f
            for activity in VALID_ACTIVITIES
            for f in sorted(glob.glob(f'{root_dir}/{activity}/*.mat'))
        ]
        if not self.data_list:
            raise FileNotFoundError(
                f'No .mat segments found under {root_dir}. See the Data section '
                'of the README for the expected folder layout.')

        # The label is the activity folder a segment sits in.
        self.category = {activity: i for i, activity in enumerate(VALID_ACTIVITIES)}
        print('category: ', self.category)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        label = self.category[sample_dir.split('/')[-2]]
        mat_data = sio.loadmat(sample_dir)

        # Node 2 runs at 20 MHz (52 subcarriers), as does node 1's NSS0;
        # only node 1's NSS1/NSS2 carry the wide 80 MHz band.
        csis, timestamps = {}, {}
        for i, (n, s) in enumerate(STREAMS):
            wide = (n == 1 and s in (1, 2))
            num = self.subcarrier_num if wide else min(self.subcarrier_num, 52)
            csis[i] = _select_subcarriers(
                mat_data[f'Node{n}_csis_NSS{s}'], n, s, num,
                self.subcarrier_motion_stats)
            timestamps[i] = mat_data[f'Node{n}_timestamps_NSS{s}'].flatten()
        return timestamps, csis, label
