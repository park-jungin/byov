
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset-name>"
    exit 1
fi

if [ "$1" = "break_eggs" ]; then
  python train.py --dataset break_eggs \
                --output_dir bestconfig  \
                --batch_size 4\
                --n_layers 12  \
                --n_layers_dec 4  \
                --freeze_base  \
                --vision_encoder_path openai/clip-vit-large-patch14  \
                --hidden_dim 1024


elif [ "$1" = "pour_milk" ]; then
  # pour milk, missing one det_bounding_box.pickle file (to be updated later), basic no object-centric encoder version below
  python train.py --dataset pour_milk \
                --output_dir bestconfig  \
                --batch_size 1\
                --n_layers 12  \
                --n_layers_dec 4  \
                --freeze_base

elif [ "$1" = "pour_liquid" ]; then
  python train.py --dataset pour_liquid \
                --output_dir bestconfig  \
                --batch_size 1\
                --n_layers 12  \
                --n_layers_dec 4  \
                --freeze_base

elif [ "$1" = "tennis_forehand" ]; then
  python train.py --dataset tennis_forehand \
                --num_frames 20 \
                --output_dir bestconfig  \
                --batch_size 1\
                --n_layers 12  \
                --n_layers_dec 4  \
                --freeze_base
else
    echo "Unknown dataset: $1, select among [break_eggs, pour_milk, pour_liquid, tennis_forehand]"
    exit 2
fi