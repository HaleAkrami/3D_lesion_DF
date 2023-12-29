#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
INPUT_DIR=$1
DATA_DIR=$2

# make the arguments mandatory and that the data dir is not a relative path
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] 
then
  echo "Usage: ./prepare_NFBS.sh <input_dir> <output_dir>"
  exit 1
fi

if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]
then
  echo "Please use absolute paths for input_dir"
  exit 1
fi


mkdir -p $DATA_DIR/v2skullstripped/NFBS/
mkdir -p $DATA_DIR/v2skullstripped/NFBS/mask

cp -r  $INPUT_DIR/t1  $DATA_DIR/v2skullstripped/NFBS/

echo "extract masks"
python get_mask.py -i $DATA_DIR/v2skullstripped/NFBS/t1 -o $DATA_DIR/v2skullstripped/NFBS/t1 -mod t1
python extract_masks.py -i $DATA_DIR/v2skullstripped/NFBS/t1 -o $DATA_DIR/v2skullstripped/NFBS/mask
python replace.py -i $DATA_DIR/v2skullstripped/NFBS/mask -s " _t1" ""

echo "Register t2"
python registration.py -i $DATA_DIR/v2skullstripped/NFBS/t1 -o $DATA_DIR/v3registered_non_iso/NFBS/t1 --modality=_t1 -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/NFBS/t1 -m $DATA_DIR/v3registered_non_iso/NFBS/mask/ -o $DATA_DIR/v3registered_non_iso_cut/NFBS/ -mode t1

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/NFBS/t1 -o $DATA_DIR/v4correctedN4_non_iso_cut/NFBS/t1 -m $DATA_DIR/v3registered_non_iso_cut/NFBS/mask

mkdir $DATA_DIR/v4correctedN4_non_iso_cut/NFBS/mask
cp $DATA_DIR/v3registered_non_iso_cut/NFBS/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/NFBS/mask
echo "Done"

# now, you should copy the files in the output directory to the data directory of the project









