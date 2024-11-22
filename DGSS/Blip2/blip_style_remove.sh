# DATASET_PATH="/workspace/hdd0/byeongcheol/Data/GTA5/images/train"
# DATASET="gtav"

# DATASET_PATH="/workspace/hdd0/byeongcheol/Data/cityscapes/leftImg8bit/val"
# DATASET="cityscapes_val"

# DATASET_PATH="/workspace/hdd0/byeongcheol/Data/BDD100k/images/val"
# DATASET="bdd100k_val"

# DATASET_PATH="/workspace/hdd0/byeongcheol/Data/Mapillary/validation/images"
# DATASET="mapillary_val"

DATASET_PATH="/workspace/hdd0/byeongcheol/Data/patches_3_flat/train/images"
DATASET="patches_3_flat_train"


STYLE_REMOVAL=""
OVERWRITE=""

for arg in "$@"; do
    if [ "$arg" == "--use_gpt_for_style_removal" ]; then
        STYLE_REMOVAL="--use_gpt_for_style_removal"
    fi
    if [ "$arg" == "--overwrite" ]; then
        STYLE_REMOVAL="--overwrite"
    fi
done

python blip_style_remove_gtav.py    \
    --dataset_path $DATASET_PATH    \
    --dataset $DATASET              \
    $STYLE_REMOVAL                   \
    $OVERWRITE



