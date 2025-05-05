import os
import numpy as np
import matplotlib.pyplot as plt
from darts.tools.hdf5_tools import load_hdf5_to_dict
from matplotlib.lines import Line2D

# --- USER SETTINGS ---
case_folder = "./results/results_ccs_fault=FM1_cut=CO1_grid=G1_top=TS2_mod=OBJ_mult=2_wbhp_000MS_Check"
#case_folder = "./results/results_ccs_grid_CCS_maarten_wbhp_000MS_Check"
output_dir = "./plot_results/single_case_plot"
os.makedirs(output_dir, exist_ok=True)

# --- Load HDF5 ---
h5_path = os.path.join(case_folder, "well_data.h5")
data = load_hdf5_to_dict(h5_path)
var_names = data['dynamic']['variable_names']
p_idx = var_names.index('pressure')
time = data['dynamic']['time']
pressure = data['dynamic']['X'][:, :, p_idx]

# --- Load Injectivity (optional) ---
injectivity = []
log_path = os.path.join(case_folder, "init_reservoir_output.log")
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            if "with WI=" in line:
                try:
                    wi_part = line.split("with WI=")[1]
                    wi_value = float(wi_part.split()[0])
                    injectivity.append(wi_value)
                except (IndexError, ValueError):
                    continue
injectivity = np.array(injectivity)

# --- Segment indexing ---
n_total = pressure.shape[1]
n_segments = (n_total - 2) // 2
res_idx = np.arange(1, 1 + n_segments)
inj_idx = np.arange(1 + n_segments + 1, 1 + 2 * n_segments + 1)

# --- Find first violation index ---
def get_violation_index(time, pressure, res_idx, inj_idx):
    for t_idx in range(len(time)):
        res_p = pressure[t_idx, res_idx]
        inj_p = pressure[t_idx, inj_idx]
        if len(res_p) == len(inj_p) and np.any(res_p > inj_p):
            return t_idx
    return None

violation_idx = get_violation_index(time, pressure, res_idx, inj_idx)

# --- Plot ---
def plot_three_timesteps(time, pressure, res_idx, inj_idx, injectivity, label_prefix,
                         output_filename, violation_idx):
    timesteps = [0, violation_idx, -1]
    titles = ['Initial', 'First Violation', 'Final']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, t_idx in enumerate(timesteps):
        if t_idx is None or t_idx >= len(time):
            axs[i].set_title(f"{titles[i]} (no violation)")
            axs[i].axis('off')
            continue

        res_p = pressure[t_idx, res_idx]
        inj_p = pressure[t_idx, inj_idx]
        segments = np.arange(len(res_p))

        ax = axs[i]
        ax.plot(segments, res_p, color='blue', label='Reservoir Pressure')
        violation_mask = res_p > inj_p
        res_p_violation = np.ma.masked_where(~violation_mask, res_p)
        ax.plot(segments, res_p_violation, color='red', label='Reservoir > Injection')
        ax.plot(segments, inj_p, marker='x', label='Injection Pressure', color='orange')

        ax.set_title(f"{titles[i]} (t = {time[t_idx]:.6f} days)")
        ax.set_xlabel("Segment Index")
        ax.grid(True)
        if i == 0:
            ax.set_ylabel("Pressure")

        if injectivity is not None and len(injectivity) >= len(segments):
            ax2 = ax.twinx()
            ax2.plot(segments, injectivity[:len(segments)],
                     linestyle='--', color='black', label='Injectivity')
            ax2.set_ylabel("Injectivity", color='black')

    custom_handles = [
        Line2D([0], [0], marker='o', color='blue', label='Reservoir Pressure', linestyle=''),
        Line2D([0], [0], marker='o', color='red', label='Reservoir > Injection', linestyle=''),
        Line2D([0], [0], marker='x', color='orange', label='Injection Pressure', linestyle=''),
        Line2D([0], [0], linestyle='--', color='black', label='Injectivity')
    ]
    fig.legend(handles=custom_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{label_prefix} Pressure Distributions Across Timesteps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_filename)
    plt.close()

plot_three_timesteps(
    time, pressure, res_idx, inj_idx, injectivity,
    "Single Case",
    os.path.join(output_dir, "single_case_pressure.png"),
    violation_idx
)

print(f"✅ Plot saved to {output_dir}")




# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from darts.tools.hdf5_tools import load_hdf5_to_dict
# from matplotlib.lines import Line2D  # Required for custom legend entry
#
# # --- USER SETTINGS ---
# non_converged_case = "006"
# converged_case = "000"
# welldata_base_dir = "./dP_4bar_welldata"
# plot_results_base = "./plot_results"
# output_folder_name = "compare_006_000"  # ← Custom folder name
# output_dir = os.path.join(plot_results_base, output_folder_name)
# os.makedirs(output_dir, exist_ok=True)
#
# def find_folder(case_number):
#     for f in os.listdir(welldata_base_dir):
#         if f"_wrate_{case_number}_" in f:
#             return os.path.join(welldata_base_dir, f)
#     return None
#
# def load_case_data(case_number):
#     folder = find_folder(case_number)
#     if folder is None:
#         raise FileNotFoundError(f"No folder found for case {case_number}")
#
#     # Load HDF5 pressure data
#     h5_path = os.path.join(folder, "well_data.h5")
#     if not os.path.exists(h5_path):
#         raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
#     data = load_hdf5_to_dict(h5_path)
#     var_names = data['dynamic']['variable_names']
#     p_idx = var_names.index('pressure')
#     time = data['dynamic']['time']
#     pressure = data['dynamic']['X'][:, :, p_idx]
#
#     # Load injectivity from log
#     injectivity = []
#     log_path = os.path.join(folder, "init_reservoir_output.log")
#     if os.path.exists(log_path):
#         with open(log_path, 'r') as f:
#             for line in f:
#                 if "with WI=" in line:
#                     try:
#                         wi_part = line.split("with WI=")[1]
#                         wi_value = float(wi_part.split()[0])
#                         injectivity.append(wi_value)
#                     except (IndexError, ValueError):
#                         continue
#     injectivity = np.array(injectivity)
#     print(f"📏 Injectivity length for case {case_number}: {len(injectivity)}")
#
#     return time, pressure, injectivity
#
# def identify_segments(pressure_array):
#     n_total = pressure_array.shape[1]
#     n_segments = (n_total - 2) // 2  # dummy + res + ghost + inj
#     res_idx = np.arange(1, 1 + n_segments)
#     inj_idx = np.arange(1 + n_segments + 1, 1 + 2 * n_segments + 1)
#     return res_idx, inj_idx
#
# def get_violation_index(time, pressure, res_idx, inj_idx):
#     for t_idx in range(len(time)):
#         res_p = pressure[t_idx, res_idx]
#         inj_p = pressure[t_idx, inj_idx]
#         if len(res_p) == len(inj_p) and np.any(res_p > inj_p):
#             return t_idx
#     return None
#
# def plot_three_timesteps(time, pressure, res_idx, inj_idx, injectivity, label_prefix,
#                          output_filename, violation_idx):
#     timesteps = [0, violation_idx, -1]
#     titles = ['Initial', 'First Violation', 'Final']
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
#
#     for i, t_idx in enumerate(timesteps):
#         if t_idx is None or t_idx >= len(time):
#             axs[i].set_title(f"{titles[i]} (no violation)")
#             axs[i].axis('off')
#             continue
#
#         res_p = pressure[t_idx, res_idx]
#         inj_p = pressure[t_idx, inj_idx]
#         segments = np.arange(len(res_p))
#
#         ax = axs[i]
#
#         # # Plot reservoir pressure: highlight violations
#         # for j, seg in enumerate(segments):
#         #     color = 'red' if res_p[j] > inj_p[j] else 'blue'
#         #     ax.plot(seg, res_p[j], marker='o', color=color)
#
#         # Plot full reservoir pressure line in blue
#         ax.plot(segments, res_p, color='blue', label='Reservoir Pressure')
#
#         # Overlay red on violating segments
#         violation_mask = res_p > inj_p
#         res_p_violation = np.ma.masked_where(~violation_mask, res_p)
#         ax.plot(segments, res_p_violation, color='red', label='Reservoir > Injection')
#
#
#         ax.plot(segments, inj_p, marker='x', label='Injection Pressure', color='orange')
#
#         ax.set_title(f"{titles[i]} (t = {time[t_idx]:.6f} days)")
#         ax.set_xlabel("Segment Index")
#         ax.grid(True)
#         if i == 0:
#             ax.set_ylabel("Pressure")
#
#         # Right y-axis for injectivity
#         if injectivity is not None and len(injectivity) >= len(segments):
#             ax2 = ax.twinx()
#             ax2.plot(segments, injectivity[:len(segments)],
#                      linestyle='--', color='black', label='Injectivity')
#             ax2.set_ylabel("Injectivity", color='black')
#
#     # Legend (manual)
#     custom_handles = [
#         Line2D([0], [0], marker='o', color='blue', label='Reservoir Pressure', linestyle=''),
#         Line2D([0], [0], marker='o', color='red', label='Reservoir > Injection', linestyle=''),
#         Line2D([0], [0], marker='x', color='orange', label='Injection Pressure', linestyle=''),
#         Line2D([0], [0], linestyle='--', color='black', label='Injectivity')
#     ]
#     fig.legend(handles=custom_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0))
#     fig.suptitle(f"{label_prefix} Pressure Distributions Across Timesteps", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.savefig(output_filename)
#     plt.close()
#
#
# # --- Load and process non-converged case ---
# time_nc, pressure_nc, injectivity_nc = load_case_data(non_converged_case)
# res_idx_nc, inj_idx_nc = identify_segments(pressure_nc)
# violation_idx_nc = get_violation_index(time_nc, pressure_nc, res_idx_nc, inj_idx_nc)
#
# plot_three_timesteps(
#     time_nc, pressure_nc, res_idx_nc, inj_idx_nc, injectivity_nc,
#     f"Non-converged Case {non_converged_case}",
#     os.path.join(output_dir, f"non_converged_case_{non_converged_case}_pressure.png"),
#     violation_idx_nc
# )
#
# # --- Load and process converged case ---
# time_c, pressure_c, injectivity_c = load_case_data(converged_case)
# res_idx_c, inj_idx_c = identify_segments(pressure_c)
#
# plot_three_timesteps(
#     time_c, pressure_c, res_idx_c, inj_idx_c, injectivity_c,
#     f"Converged Case {converged_case}",
#     os.path.join(output_dir, f"converged_case_{converged_case}_pressure.png"),
#     violation_idx_nc  # use same t_idx as non-converged
# )
#
# print(f"✅ Saved plots in: {output_dir}")
#
