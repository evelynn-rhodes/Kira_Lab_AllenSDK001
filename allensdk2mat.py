from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import scipy.io as sio
import numpy as np
import pandas as pd
import os

# --- Load experiment data ---
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
container_id = 511510917

exp = boc.get_ophys_experiments(
    experiment_container_ids=[container_id],
    stimuli=[stim_info.STATIC_GRATINGS]
)[0]
dataset = boc.get_ophys_experiment_data(exp['id'])

# --- Get all cells ---
cell_ids = dataset.get_cell_specimen_ids()

# --- Get metadata and sort cells ---
cell_metadata = pd.DataFrame(boc.get_cell_specimens())
cell_metadata = cell_metadata[cell_metadata["cell_specimen_id"].isin(cell_ids)]

containers = pd.DataFrame(boc.get_experiment_containers())
cell_metadata = cell_metadata.merge(
    containers[["id", "targeted_structure"]],
    left_on="experiment_container_id",
    right_on="id",
    how="left"
).drop(columns="id")

cell_metadata = cell_metadata.sort_values(["targeted_structure", "cell_specimen_id"])
sorted_cell_ids = cell_metadata["cell_specimen_id"].tolist()

# --- Extract data (load all, then reorder) ---
timestamps, dff_traces_all = dataset.get_dff_traces()
orig_cell_ids = dataset.get_cell_specimen_ids()
id_to_idx = {cid: i for i, cid in enumerate(orig_cell_ids)}
order_idx = np.array([id_to_idx[cid] for cid in sorted_cell_ids], dtype=int)
dff_traces_sorted = dff_traces_all[order_idx, :]

# --- Stimulus table ---
stim_table = dataset.get_stimulus_table("static_gratings")

# --- Save to 'data' folder ---
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "allen_dataset.mat")

sio.savemat(output_path, {
    "timestamps": timestamps,
    "sorted_cell_ids": np.array(sorted_cell_ids),
    "dff_traces_sorted": dff_traces_sorted,
    "stim_table": stim_table.to_dict("list"),
    "cell_metadata": cell_metadata.to_dict("list")
})

print(f"Saved to {output_path}")
