#!/bin/bash
OBJECTS=("s0126_heart" "s0002_kidney_left" "s0468_lung_left" "s0564_vertebrae")
#OBJECTS=("s0468_lung_left")
#OBJECTS=("s0126_heart")
#OBJECTS=("s0564_vertebrae")

for OBJECT in "${OBJECTS[@]}"; do
    DIR="$OBJECT"
    DATA="path_to_your_input_data_folder/{$OBJECT}"

    echo "====================================================="
    echo " Processing: $OBJECT"
    echo " Data path : $DATA"
    echo " Output dir: $DIR"
    echo "====================================================="

    # Check dataset path exists
    if [[ ! -d "$DATA" ]]; then
        echo "Error: dataset folder not found: $DATA"
        continue
    fi

    #Clean old run
    rm -rf "$DIR"
  
    #  Train
    python train.py -s "$DATA" -m "$DIR" --poly_degree 2 --pipeline seg

    # Render
    echo "Rendering..."
    python render.py --model_path "$DIR/" --interp 8

    # Make video
    python video.py \
        --input_folder "$DIR/render" \
        --output_folder "$DIR" \
        --output_name "video.mp4" \
        --fps 15

    echo "Finished: $OBJECT (results in $DIR/video.mp4)"
done

