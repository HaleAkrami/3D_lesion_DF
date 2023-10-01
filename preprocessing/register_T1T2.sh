#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
INPUT_DIR=$1
DATA_DIR=$2
T2_DIR=$3

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




echo "Register t2"
python registrationT1.py -i $T2_DIR/v4correctedN4_non_iso_cut/Brats21/t2 -o $DATA_DIR/v5registered_non_iso/Brats21/t2 --modality=_t2  -templ $INPUT_DIR/v4correctedN4_non_iso_cut/Brats21/t1


echo "Done"


# now, you should copy the files in the output directory to the data directory of the project









