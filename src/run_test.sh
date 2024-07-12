#!/bin/bash
python train.py --epochs 100 --net conv-net --lr 1e-4
python train.py --epochs 100 --net conv-net --augment_data --lr 1e-4

python train.py --epochs 100 --net res-net-18 --lr 1e-4
python train.py --epochs 100 --net res-net-18 --augment_data --lr 1e-4

python train.py --epochs 100 --net res-net-34 --lr 1e-4
python train.py --epochs 100 --net res-net-34 --augment_data --lr 1e-4

python train.py --epochs 100 --net res-net-18-pre --lr 1e-5
python train.py --epochs 100 --net res-net-18-pre --augment_data --lr 1e-5

python train.py --epochs 100 --net res-net-34-pre --lr 1e-5
python train.py --epochs 100 --net res-net-34-pre --augment_data --lr 1e-5
