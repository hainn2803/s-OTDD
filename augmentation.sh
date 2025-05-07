PARENT_DIR="saved_augmentation"

for seed in {1..15}
do
    echo "Running with seed $seed..."
    python3 augmentation.py --parent_dir "$PARENT_DIR" --seed $seed
done