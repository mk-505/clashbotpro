import sys
from pathlib import Path
sys.path.append(str(Path("D:/RL_Finance/KataCR")))



from katacr.build_dataset.generator import Generator
from pathlib import Path
from PIL import Image
import numpy as np
import os
import argparse

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Generate synthetic Clash Royale dataset.")
parser.add_argument("--n", type=int, default=10, help="Number of datapoints to generate")
parser.add_argument("--seed", type=int, default=35, help="seed number for the generator")
parser.add_argument("--num_obj", type=int, default=15, help="Number of Object we wouldl like to generate in the image")
args = parser.parse_args()

# ----------------------------
# Setup
# ----------------------------
avail_names = [
    'king-tower', 'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower',
    'dagger-duchess-tower-bar', 'tower-bar', 'king-tower-bar', 'bar',
    'bar-level', 'clock', 'emote', 'elixir', 'goblin-demolisher',
    'suspicious-bush', 'archer'
]

generator = Generator(
    seed=args.seed,
    intersect_ratio_thre=0.5,
    augment=True,
    map_update={'mode': 'dynamic', 'size': 5},
    noise_unit_ratio=1/4,
    avail_names=avail_names
)

path_argument_ds = Path('d:/RL_Finance/Clash-Royale-Dataset/images/part2')
path_generation = path_argument_ds / "xxy_argument_ds"
path_generation.mkdir(exist_ok=True)

# ----------------------------
# Function to Save Boxes
# ----------------------------
def save_boxes_to_txt(boxes: np.ndarray, img_filename: str, output_dir: str):
    base_name = os.path.splitext(img_filename)[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(txt_path, "w") as f:
        for box in boxes:
            x_center, y_center, width, height, side, class_id = box
            line = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {int(side)} 0 0 0 0 0 0\n"
            f.write(line)

# ----------------------------
# Generation Loop
# ----------------------------
for i in range(args.n):
    generator.add_tower()
    generator.add_unit(n=args.num_obj)

    # Generate image with boxes
    img_with_box_path = path_generation / f"test{i}.jpg"
    x, box, _ = generator.build(verbose=False, show_box=False, save_path=str(img_with_box_path))
    save_boxes_to_txt(box, img_with_box_path.name, str(path_generation))


    print(f"Generated pair {2*i}, {2*i+1} | Box count: {box.shape[0]}")
    generator.reset()



## Run the following command to generate the dataset

## python katacr/build_dataset/generate_argumented_ds.py --n 100 --seed 42 --num_obj 10