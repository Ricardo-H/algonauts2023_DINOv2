#!/bin/bash

mkdir result
for subj in data/*
do
    subj_name=$(basename $subj)
    mkdir result/$subj_name
    cp $subj/test_split/test_fmri/lh_pred_test.npy result/$subj_name
    cp $subj/test_split/test_fmri/rh_pred_test.npy result/$subj_name
done
cd result
zip -r result.zip *
cd ..
mv result/result.zip .
rm -r result