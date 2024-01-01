#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
INPUT_DIR=$1
DATA_DIR=$2

# make the arguments mandatory and that the data dir is not a relative path
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] 
then
  echo "Usage: ./prepare_MSLUB.sh <input_dir> <output_dir>"
  exit 1
fi

if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]
then
  echo "Please use absolute paths for input_dir"
  exit 1
fi  

echo "Resample"
mkdir -p $DATA_DIR/v1resampled/ATLAS/t1
python resample.py -i $INPUT_DIR/t1 -o $DATA_DIR/v1resampled/ATLAS/t1 -r 1.0 1.0 1.0 
## rename files for standard naming
for file in $DATA_DIR/v1resampled/ATLAS/t1/*
do
  mv "$file" "${file%_T1W.nii.gz}_t1.nii.gz"
done

echo "Generate masks"
# mkdir -p $DATA_DIR/v2skullstripped/MSLUB/t2
CUDA_VISIBLE_DEVICES=0 hd-bet -i $DATA_DIR/v1resampled/ATLAS/t1 -o $DATA_DIR/v2skullstripped/ATLAS/t1 # --overwrite_existing=0
python extract_masks.py -i $DATA_DIR/v2skullstripped/ATLAS/t1 -o $DATA_DIR/v2skullstripped/ATLAS/mask
python replace.py -i $DATA_DIR/v2skullstripped/ATLAS/mask -s " _t1" ""

# copy segmentation masks to the data directory
mkdir -p $DATA_DIR/v2skullstripped/ATLAS/seg
cp -r $INPUT_DIR/seg/* $DATA_DIR/v2skullstripped/ATLAS/seg/

for file in $DATA_DIR/v2skullstripped/ATLAS/seg/*
do
  mv "$file" "${file%consensus_gt.nii.gz}seg.nii.gz"
done


echo "Register t2"
python registration.py -i $DATA_DIR/v2skullstripped/ATLAS/t1 -o $DATA_DIR/v3registered_non_iso/ATLAS/t1 --modality=_t1 -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/ATLAS/t1 -m $DATA_DIR/v3registered_non_iso/ATLAS/mask/ -o $DATA_DIR/v3registered_non_iso_cut/ATLAS/ -mode t1

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/ATLAS/t1 -o $DATA_DIR/v4correctedN4_non_iso_cut/ATLAS/t1 -m $DATA_DIR/v3registered_non_iso_cut/ATLAS/mask
mkdir $DATA_DIR/v4correctedN4_non_iso_cut/ATLAS/mask
cp $DATA_DIR/v3registered_non_iso_cut/ATLAS/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/ATLAS/mask
mkdir $DATA_DIR/v4correctedN4_non_iso_cut/ATLAS/seg
cp $DATA_DIR/v3registered_non_iso_cut/ATLAS/seg/* $DATA_DIR/v4correctedN4_non_iso_cut/ATLAS/seg
echo "Done"


# now, you should copy the files in the output directory to the data directory of the project









