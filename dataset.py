import os
import jax
import json
import numpy as np
from time import time
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader
from models.pointnet2_utils import pc_normalize


class PartNormalDataset(Dataset):

    def __init__(
        self,
        root="shapenetcore_partanno_segmentation_benchmark_v0_normal/",
        npoints=2500,
        split="train",
        class_choice=None,
        normal_channel=False,
    ):
        """
        Creates a dataset object for the PartNormalDataset.

        Args
        ----
            root: str
                The root directory of the dataset.
                (default='shapenetcore_partanno_segmentation_benchmark_v0_normal')
            npoints: int
                The number of points to sample from each point cloud.
                (default=2500)
            split: str
                The split of the dataset to use. One of 'train', 'test', 'val', 'trainval'.
                (default='train')
            class_choice: list
                A list of classes to use. If None, all classes are used.
                (default=None)
            normal_channel: bool
                Whether to include normal information in the point cloud.
                (default=False)
        """

        data_dir = os.environ.get("SCRATCH") # scratch is the directory for fast i/o
        if data_dir is None:
            raise ValueError("Please set the DATA environment variable.")
        self.root = os.path.join(data_dir, root)
        self.npoints = npoints
        self.catfile = os.path.join(self.root, "synsetoffset2category.txt")

        self.key = jax.random.PRNGKey(0)  # for random selection of points

        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, "r") as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # self.cat = {k: v for k, v in self.cat.items()} # redundant?

        # create a class to index mapping
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        # map classes to the ones we want to use
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}

        # load train, val, and test splits
        with open(
            os.path.join(
                self.root, "train_test_split", "shuffled_train_file_list.json"
            ),
            "r",
        ) as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])

        with open(
            os.path.join(self.root, "train_test_split", "shuffled_val_file_list.json"),
            "r",
        ) as f:
            val_ids = set([str(d.split("/")[2]) for d in json.load(f)])

        with open(
            os.path.join(self.root, "train_test_split", "shuffled_test_file_list.json"),
            "r",
        ) as f:
            test_ids = set([str(d.split("/")[2]) for d in json.load(f)])

        # iterate over the categories and load the data
        # for each category given a split
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == "trainval":
                fns = [
                    fn
                    for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == "train":
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print("Unknown split: %s. Exiting.." % (split))
                exit(-1)

            # create a list of actual files to load per class
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]  # again redundant?
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        # this is just a list of tuples of the form (category, filename)
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        # self.classes is a redundant mapping from category ('Chair') to an int (0) ????
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {
            "Earphone": [16, 17, 18],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Rocket": [41, 42, 43],
            "Car": [8, 9, 10, 11],
            "Laptop": [28, 29],
            "Cap": [6, 7],
            "Skateboard": [44, 45, 46],
            "Mug": [36, 37],
            "Guitar": [19, 20, 21],
            "Bag": [4, 5],
            "Lamp": [24, 25, 26, 27],
            "Table": [47, 48, 49],
            "Airplane": [0, 1, 2, 3],
            "Pistol": [38, 39, 40],
            "Chair": [12, 13, 14, 15],
            "Knife": [22, 23],
        }

        # cache for the dataset! interesting...
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):

        # check cache first
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:

            # get the category and filename
            cat, fn = self.datapath[index]
            cls = self.classes[cat]
            cls = np.array([cls], dtype=np.int32)
            data = np.loadtxt(fn, dtype=np.float32)# load the point cloud data

            # pick surface normals or not
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]

            # the last column is the segmentation label for the part
            seg = data[:, -1].astype(np.int32)

            # cache the data
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        # normalize the point cloud
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # split the key and sample the points
        choice = np.random.choice(len(seg), self.npoints, replace=True)
                
        # select
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


def collate_fn(batch):
    """
    Collates a batch of data.

    Args
    ----
        batch: list
            A list of tuples of the form (point_set, cls, seg).

    Returns
    -------
        points: jnp.array
            A batch of point clouds.
        cls: jnp.array
            A batch of class labels.
        seg: jnp.array
            A batch of segmentation labels.
    """

    point_list, cls_list, seg_list = zip(*batch)

    points = jnp.array(point_list)
    cls = jnp.array(cls_list)
    seg = jnp.array(seg_list)
    
    return points, cls, seg


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
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


if __name__ == "__main__":
    data = PartNormalDataset(split="train", normal_channel=False)
    dataloader = JAXDataLoader(data, batch_size=4, shuffle=True)
    init = time()
    for point, label, seg in dataloader:
        cur = time()
        print(f"Time: {cur - init:.2f}s")
        # print(point.shape)
        # print(type(point))
        # print(label.shape)
        init = time()
