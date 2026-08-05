#!/bin/bash

# Compares flow matching (FM) conditional flow matching (CFM) and optimal
# flowTransport conditional flow matching on four datasets.  twodim is not possible
# flowFor the flow matching algorithm as it has a non-gaussian source distribution.
# FM is therefore only run on three datasets.
python src/flowTrain.py -m experiment=cfm \
  flowModel=cfm,otcfm \
  launcher=mila_cpu_cluster \
  flowModel.sigma_min=0.1 \
  datamodule=scurve,moons,twodim,gaussians \
  seed=42,43,44,45,46 &

# Sleep to avoid launching jobs at the same time
sleep 1
python src/flowTrain.py -m experiment=cfm \
  flowModel=fm \
  launcher=mila_cpu_cluster \
  flowModel.sigma_min=0.1 \
  datamodule=scurve,moons,gaussians \
  seed=42,43,44,45,46 &


