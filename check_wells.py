from darts.tools.hdf5_tools import load_hdf5_to_dict

dir = r"Y:\Desktop\scratch-main-maarten_files\maarten_files\results\results_ccs_grid_CCS_maarten_wrate_000_h5\well_data.h5"

results = load_hdf5_to_dict(dir)

var_names = results['dynamic']['variable_names']
pressure_index = var_names.index('pressure')

X = results['dynamic']['X']  # shape likely (time, wells, variables)
pressure = X[:, :, pressure_index]  # 2D array: (time_steps, wells)

import matplotlib.pyplot as plt

time = results['dynamic']['time']
plt.plot(time, pressure[:, 0])  # pressure over time for well 0
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.title("Pressure vs Time for Well 0")
plt.grid(True)
plt.show()

# Index 1–31 for reservoir, 32–62 for injection
reservoir_pressure = pressure[:, 1:32]  # shape (45, 31)
injection_pressure = pressure[:, 32:63]  # shape (45, 31)

time_step = 0  # adjust as needed
res_p = reservoir_pressure[time_step, :]
inj_p = injection_pressure[time_step, :]

import matplotlib.pyplot as plt
import numpy as np

segment_ids = np.arange(31)

plt.plot(segment_ids, res_p, marker='o', label='Reservoir Pressure')
plt.plot(segment_ids, inj_p, marker='x', label='Injection Pressure')

plt.xlabel("Segment Index")
plt.ylabel("Pressure")
plt.title(f"Injection vs Reservoir Pressure at Time Step {time_step}")
plt.legend()
plt.grid(True)
plt.show()

overpressure = inj_p > res_p
for i, over in enumerate(overpressure):
    if over:
        print(f"Segment {i}: Injection pressure ({inj_p[i]:.2f}) > Reservoir pressure ({res_p[i]:.2f})")



print(results)

print('hello')