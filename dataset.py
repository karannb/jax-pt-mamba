import os
import json
import torch
import numpy as np
from time import time
from torch.utils.data import Dataset, DataLoader


# Helper functions
def pc_normalize(pc):
    """
    Normalize point cloud data.

    Args:
        pc (numpy.ndarray): Point cloud data, [N, 3]

    Returns:
        numpy.ndarray: Normalized point cloud data, [N, 3]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


# Dataset class
class ShapenetPartDataset(Dataset):
    def __init__(
        self, split="train", class_choice=None, normal_channel=False, num_points=2048
    ):
        self.root = os.path.join(
            os.environ.get("SCRATCH"),
            "shapenetcore_partanno_segmentation_benchmark_v0_normal",
        )
        self.split = split
        self.class_choice = class_choice
        self.normal_channel = normal_channel
        self.num_points = num_points
        self.catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}
        self.meta = {}
        self.cache = {}
        self.cache_size = 80000

        self._load_categories()
        self._load_metadata()
        self._prepare_data_paths()
        self._map_classes()

    def _load_categories(self):
        try:
            with open(self.catfile, "r") as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]
            self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        except FileNotFoundError:
            raise Exception(f"Category file not found at {self.catfile}")

        if self.class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}

    def _load_metadata(self):
        try:
            with open(
                os.path.join(
                    self.root, "train_test_split", "shuffled_train_file_list.json"
                ),
                "r",
            ) as f:
                train_ids = set([str(d.split("/")[2]) for d in json.load(f)])

            with open(
                os.path.join(
                    self.root, "train_test_split", "shuffled_val_file_list.json"
                ),
                "r",
            ) as f:
                val_ids = set([str(d.split("/")[2]) for d in json.load(f)])

            with open(
                os.path.join(
                    self.root, "train_test_split", "shuffled_test_file_list.json"
                ),
                "r",
            ) as f:
                test_ids = set([str(d.split("/")[2]) for d in json.load(f)])

        except FileNotFoundError:
            raise Exception("Train/test/val split files not found")

        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if self.split == "trainval":
                fns = [
                    fn
                    for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif self.split == "train":
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif self.split == "val":
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif self.split == "test":
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

    def _prepare_data_paths(self):
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

    def _map_classes(self):
        self.classes = {i: self.classes_original[i] for i in self.cat.keys()}
        self.seg_classes = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }
        self.identity = np.eye(50)

    def __getitem__(self, index):
        # object is cached in memory
        if index in self.cache:
            point_set, object_label, seg = self.cache[index]

        # read the object from disk
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            object_label = int(self.classes[cat])
            data = np.loadtxt(fn[1]).astype(np.float32)

            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]

            seg = data[:, -1].astype(np.int32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, object_label, seg)

        # normalize x, y and z coordinates
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # select num_points points from the sequence with replacement
        choice = np.random.choice(len(seg), self.num_points, replace=True)

        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        # # Sort the points according to the key (x_coord, y_coord, z_coord)
        # sorted_indices = np.lexsort((point_set[:, 2], point_set[:, 1], point_set[:, 0]))
        # point_set = point_set[sorted_indices, 1:]

        # seg = seg[sorted_indices]
        # seg = self.identity[seg]

        return point_set, object_label, seg

    def __len__(self):
        return len(self.datapath)


def numpy_collate_fn(batch):
    points, object_labels, segmentation_labels = zip(*batch)

    object_labels = np.array(object_labels)

    tokens = np.array(points)
    targets = np.array(segmentation_labels)

    # timesteps = tokens[:, :, 0]
    # timesteps = np.diff(timesteps, axis=1, append=timesteps[:, -1:])
    # lengths = np.array([len(seq) for seq in tokens])[..., None]
    return tokens, object_labels, targets  # , timesteps, lengths


class JAXDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


# Main function
def main():
    # train_loader, _, _ = dataloader_generator(seed=0)
    train_data = ShapenetPartDataset(
        (
            "/p/project1/eelsaisdc/bania1/data/"
            "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        )
    )
    train_loader = JAXDataLoader(train_data, batch_size=4, shuffle=True)

    n = len(train_loader)
    iterator = iter(train_loader)
    end = None
    for i in range(n):
        start = time()
        batch = next(iterator)
        if end != None:
            print(f"Time taken for batch {i+1}: {start - end:.4f} seconds")
        end = time()


if __name__ == "__main__":
    main()
