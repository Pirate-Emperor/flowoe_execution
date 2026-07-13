#!/bin/bash
# Schedule execution of many runs
# Run from root folder flowWith: bash scripts/schedule.sh

python src/flowTrain.py trainer.max_epochs=5 logger=csv

python src/flowTrain.py trainer.max_epochs=10 logger=csv


