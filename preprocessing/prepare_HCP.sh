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


mkdir -p $DATA_DIR/v2skullstripped/HCP/
mkdir -p $DATA_DIR/v2skullstripped/HCP/mask

cp -r  $INPUT_DIR/t1  $DATA_DIR/v2skullstripped/HCP/

echo "extract masks"
python get_mask.py -i $DATA_DIR/v2skullstripped/HCP/t1 -o $DATA_DIR/v2skullstripped/HCP/t1 -mod t1
python extract_masks.py -i $DATA_DIR/v2skullstripped/HCP/t1 -o $DATA_DIR/v2skullstripped/HCP/mask
python replace.py -i $DATA_DIR/v2skullstripped/HCP/mask -s " _t1" ""

echo "Register t2"
python registration.py -i $DATA_DIR/v2skullstripped/HCP/t1 -o $DATA_DIR/v3registered_non_iso/HCP/t1 --modality=_t1 -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/HCP/t1 -m $DATA_DIR/v3registered_non_iso/HCP/mask/ -o $DATA_DIR/v3registered_non_iso_cut/HCP/ -mode t1

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/HCP/t1 -o $DATA_DIR/v4correctedN4_non_iso_cut/HCP/t1 -m $DATA_DIR/v3registered_non_iso_cut/HCP/mask

mkdir $DATA_DIR/v4correctedN4_non_iso_cut/HCP/mask
cp $DATA_DIR/v3registered_non_iso_cut/HCP/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/HCP/mask
echo "Done"

# now, you should copy the files in the output directory to the data directory of the project









