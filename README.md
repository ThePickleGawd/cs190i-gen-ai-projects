# CS 190I: Generative AI and Deep Learning

Projects for Spring '25 CS 190I

## Prerequisites

```
pip install uv
```

## Running Batches on CSIL

```bash
# Connect to GPU
srun --gpus=1 --nodes=1 --time=24:00:00 --cpus-per-task=4 --pty bash

# See job queue
squeue | grep dylanlu

# Cancel job ID, find with squeue
scancel [job-id]

# Keep terminal alive
tmus new -s example
tmux ls
tmux a -t example # Attach

# Run batch
sbatch sbatch_script.sh
```
