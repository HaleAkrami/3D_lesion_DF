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


mkdir -p $DATA_DIR/v2skullstripped/CamCan/
mkdir -p $DATA_DIR/v2skullstripped/CamCan/mask

cp -r  $INPUT_DIR/t1  $DATA_DIR/v2skullstripped/CamCan/
cp -r  $INPUT_DIR/t2  $DATA_DIR/v2skullstripped/CamCan/

echo "extract masks"
python get_mask.py -i $DATA_DIR/v2skullstripped/CamCan/t1 -o $DATA_DIR/v2skullstripped/CamCan/t1 -mod t1
python extract_masks.py -i $DATA_DIR/v2skullstripped/CamCan/t1 -o $DATA_DIR/v2skullstripped/CamCan/mask
python replace.py -i $DATA_DIR/v2skullstripped/CamCan/mask -s " _t1" ""


echo "Register t2"
python registration_v2.py -i $DATA_DIR/v2skullstripped/CamCan/t1 -o $DATA_DIR/v3registered_non_iso/CamCan/t1 --modality=_t1 -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Done"

# now, you should copy the files in the output directory to the data directory of the project









