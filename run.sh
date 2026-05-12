#!/bin/bash

# Sweep n from 1 to 10
for n in {1..10}
do
    echo "Running for n = $n"
    python3 test.py $n
done