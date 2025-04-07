#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Optional: Activate a virtual environment (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 256 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3



# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 257 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3


# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 258 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 259 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3

# Running multiple Python scripts
echo "Running script1.py..."
python solve_vec_network_vmap.py --seed 260 --batch_size 128 --rf_num 1024 --rf_sigma 0.0 --max_timesteps 5e3




echo "All scripts executed successfully."
