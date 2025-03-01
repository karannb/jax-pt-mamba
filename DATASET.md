## Dataset

The overall directory structure should be:
```
│PointMamba/
├──cfgs/
├──data/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──.......
```
Or store the dataset in a particular directory and change the root path in `dataset.py`.

### ShapeNetPart Dataset:

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 
