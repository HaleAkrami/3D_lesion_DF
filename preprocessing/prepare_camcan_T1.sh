#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
INPUT_DIR=$1
DATA_DIR=$2

# make the arguments mandatory and that the data dir is not a relative path
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] 
then
  echo "Usage: ./prepare_IXI.sh <input_dir> <output_dir>"
  exit 1
fi

if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]
then
  echo "Please use absolute paths for input_dir"
  exit 1
fi

echo "Resample"
mkdir -p $DATA_DIR/v1resampled/CamCan/t1
python resample.py -i $INPUT_DIR/t1 -o $DATA_DIR/v1resampled/CamCan/t1 -r 1.0 1.0 1.0 
# rename files for standard naming
for file in $DATA_DIR/v1resampled/CamCan/t1/*
do
  mv "$file" "${file%-T1.nii.gz}_t1.nii.gz"
done

echo "Generate masks"
CUDA_VISIBLE_DEVICES=0 hd-bet -i $DATA_DIR/v1resampled/CamCan/t1 -o $DATA_DIR/v2skullstripped/CamCan/t1 
python extract_masks.py -i $DATA_DIR/v2skullstripped/CamCan/t1 -o $DATA_DIR/v2skullstripped/CamCan/mask
python replace.py -i $DATA_DIR/v2skullstripped/CamCan/mask -s " _t1" ""

echo "Register t1"
python registration.py -i $DATA_DIR/v2skullstripped/CamCan/t1 -o $DATA_DIR/v3registered_non_iso/CamCan/t1 --modality=_t1 -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/CamCan/t1 -m $DATA_DIR/v3registered_non_iso/CamCan/mask/ -o $DATA_DIR/v3registered_non_iso_cut/CamCan/ -mode t1

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/CamCan/t1 -o $DATA_DIR/v4correctedN4_non_iso_cut/CamCan/t1 -m $DATA_DIR/v3registered_non_iso_cut/CamCan/mask

mkdir $DATA_DIR/v4correctedN4_non_iso_cut/CamCan/mask
cp $DATA_DIR/v3registered_non_iso_cut/CamCan/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/CamCan/mask
echo "Done"

# now, you should copy the files in the output directory to the data directory of the project









