from pathlib import Path
import imageio
from natsort import natsorted
from tqdm.auto import tqdm

# Configuration
fps = 10
outs_dir = Path("outs/outs_MNIST_20_50_26_451187__25_03_24")
timelapse_dir = "snapshots"
# distrib_type = "SUMMED"  # "SUMMED" or "COMBINED"
distrib_type = "COMBINED"  # "SUMMED" or "COMBINED"

# Output
output_dir = Path("gifs")
# Use the entire directory name in the save name
save_name = f"{outs_dir.name}_{distrib_type}.gif"

output_dir.mkdir(exist_ok=True)

# Build search pattern and collect sorted image paths
search_pat = outs_dir / timelapse_dir / f"pi_marginal_{distrib_type}_*.png"
image_paths = natsorted(list(search_pat.parent.glob(search_pat.name)))

# Read and collect images
imgs = [
    imageio.imread(str(img_path))
    for img_path in tqdm(image_paths, desc="Reading images")
]

# Save to GIF with indefinite looping
imageio.mimsave(output_dir / save_name, imgs, fps=fps, loop=0)
print(f"GIF saved to {output_dir / save_name}")
