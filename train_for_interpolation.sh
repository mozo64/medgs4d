#!/bin/bash
OBJECT="prostate"
DATA="path_to_your_input_data_folder/{$OBJECT}"
DIR="path_to_your_output_folder/{$OBJECT}"
output_name=video.mp4

rm -r $DIR
python3 train.py -s $DATA -m $DIR --poly_degree 7
python3 render.py --model_path $DIR/ --interp 2
python video.py --input_folder $DIR/render --output_folder $DIR --output_name $output_name