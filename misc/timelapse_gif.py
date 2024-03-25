from pathlib import Path
import imageio
from natsort import natsorted
from tqdm.auto import tqdm
import re  # For parsing filenames to extract epochs

# Configuration
fps = 10
max_epochs = -1  # Set to -1 to include all epochs
outs_dir = Path("outs/outs_MNIST_20_50_26_451187__25_03_24")
timelapse_dir = "snapshots"
distrib_type = "COMBINED"  # "SUMMED" or "COMBINED"

# Output
output_dir = Path("gifs")
# Adjust save name based on whether a max_epochs limit is applied
if max_epochs == -1:
    save_name = f"{outs_dir.name}_{distrib_type}.gif"
else:
    save_name = f"{outs_dir.name}_{distrib_type}_up_to_epoch_{max_epochs}.gif"

output_dir.mkdir(exist_ok=True)

# Build search pattern and collect sorted image paths
search_pat = outs_dir / timelapse_dir / f"pi_marginal_{distrib_type}_*.png"
image_paths = natsorted(list(search_pat.parent.glob(search_pat.name)))

if max_epochs == -1:
    filtered_paths = image_paths
else:
    # Filter paths to include only those up to the specified max_epochs
    filtered_paths = []
    for path in image_paths:
        match = re.search(r"epoch_(\d+)", path.stem)
        if match and int(match.group(1)) <= max_epochs:
            filtered_paths.append(path)

# Read and collect images
imgs = [
    imageio.imread(str(img_path))
    for img_path in tqdm(filtered_paths, desc="Reading images")
]

# Save to GIF with indefinite looping
imageio.mimsave(output_dir / save_name, imgs, fps=fps, loop=0)
print(f"GIF saved to {output_dir / save_name}")
