#!/bin/bash

for i in {1..10}
do
  python3 ga_experiment.py --run "$i" &
done