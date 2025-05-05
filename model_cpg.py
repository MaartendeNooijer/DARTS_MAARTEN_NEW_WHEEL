# from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube

from cpg_reservoir import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube

from darts.discretizer import load_single_float_keyword
from darts.engines import value_vector

from darts.tools.gen_cpg_grid import gen_cpg_grid

# from darts.models.cicd_model import CICDModel
from cicd_model import CICDModel
from darts.engines import value_vector, index_vector

import numpy as np


def fmt(x):
    return '{:.3}'.format(x)


#####################################################

class Model_CPG(CICDModel):
    def __init__(self):
        super().__init__()

    def init_reservoir(self):
        if self.idata.generate_grid:
            if self.idata.grid_out_dir is None:
                self.idata.gridname = None
                self.idata.propname = None
            else:  # save generated grid to grdecl files
                os.makedirs(self.idata.grid_out_dir, exist_ok=True)
                self.idata.gridname = os.path.join(self.idata.grid_out_dir, 'grid.grdecl')
                self.idata.propname = os.path.join(self.idata.grid_out_dir, 'reservoir.in')
                arrays = gen_cpg_grid(nx=self.idata.geom.nx, ny=self.idata.geom.ny, nz=self.idata.geom.nz,
                                      dx=self.idata.geom.dx, dy=self.idata.geom.dy, dz=self.idata.geom.dz,
                                      start_z=self.idata.geom.start_z,
                                      permx=self.idata.rock.permx, permy=self.idata.rock.permy,
                                      permz=self.idata.rock.permz,
                                      poro=self.idata.rock.poro,
                                      gridname=self.idata.gridname, propname=self.idata.propname)
        else:
            # read grid and rock properties
            arrays = read_arrays(self.idata.gridfile, self.idata.propfile)
            check_arrays(arrays)
            # Replace zero porosity values with a small number to avoid inactive cells and numerical issues
            poro_array = arrays.get('PORO')
            if poro_array is not None:
                poro_array[poro_array <= 0] = self.idata.geom.min_poro
                arrays['PORO'] = poro_array

        if self.idata.geom.burden_layers > 0:
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=self.idata.geom.burden_layers,
                               initial_thickness=self.idata.geom.burden_init_thickness,
                               property_dictionary=arrays,
                               burden_layer_prop_value=self.idata.rock.burden_prop)

        self.reservoir = CustomCPGReservoir(self.timer, arrays, faultfile=self.idata.faultfile,
                                            minpv=self.idata.geom.minpv)  # NEW #was self.reservoir = CPG_Reservoir(self.timer, arrays, minpv=self.idata.geom.minpv)
        self.reservoir.discretize()

        # helps reading in satnum values from the properties file and store it as op_num
        from darts.discretizer import index_vector as index_vector_discr, load_single_int_keyword
        opnum_cpp = index_vector_discr()  # self.discr_mesh.coord
        load_single_int_keyword(opnum_cpp, self.idata.propfile, 'SATNUM', -1)

        arrays['SATNUM'] = np.array(opnum_cpp, copy=False)

        opnum_full = arrays["SATNUM"]

        # Get the mapping of active cells #NEW
        global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)  # NEW

        # Select only active cell SATNUM values #NEW
        opnum_active = opnum_full[global_to_local >= 0]  # NEW

        self.reservoir.op_num = opnum_active - 1

        # Assign this condensed op_num to the mesh
        self.reservoir.mesh.op_num = index_vector(self.reservoir.op_num)

        # Update global_data for satnum and op_num  #NEW
        self.reservoir.global_data.update({'op_num': self.reservoir.op_num})

        # store modified arrrays (with burden layers) for output to grdecl
        self.reservoir.input_arrays = arrays

        volume = np.array(self.reservoir.mesh.volume, copy=False)
        poro = np.array(self.reservoir.mesh.poro, copy=False)
        self.total_poro_volume = sum(volume[:self.reservoir.mesh.n_blocks] * poro)
        print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))

        # imitate open-boundaries with a large volume
        bv = self.idata.geom.bound_volume  # volume, will be assigned to each boundary cell [m3]
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        mask_shale = (self.reservoir.op_num == 2)  # & (global_to_local >= 0)
        mask_sand = ((self.reservoir.op_num == 0) | (self.reservoir.op_num == 1))  # & (global_to_local >= 0)

        self.reservoir.conduction[mask_shale] = self.idata.rock.conduction_shale
        self.reservoir.conduction[mask_sand] = self.idata.rock.conduction_sand

        self.reservoir.hcap[mask_shale] = self.idata.rock.hcap_shale
        self.reservoir.hcap[mask_sand] = self.idata.rock.hcap_sand

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)

        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap.copy(), l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction.copy(), l2g,
                                                                             g2l)})
        # 'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l) })

        self.set_physics()

        # time stepping and convergence parameters
        sim = self.idata.sim  # short name
        self.set_sim_params(first_ts=sim.first_ts, mult_ts=sim.mult_ts, max_ts=sim.max_ts, runtime=sim.runtime,
                            tol_newton=sim.tol_newton, tol_linear=sim.tol_linear)
        if hasattr(sim, 'linear_type'):
            self.params.linear_type = sim.linear_type

        self.timer.node["initialization"].stop()
        self.init_tracked_cells()
        # self.check_conductivity()

    def init_tracked_cells(self):

        # Define grid cells you want to monitor (1-based IJK)
        # self.ijk_to_track = [ ] #[(92, 112, 38)] # test injector 92, 94 (midden grid) #gilles cells for tracking with prd = 55, 67 & inj 132, 92 => [(131, 77, 39), (131, 76, 34), (42, 53, 31), (42, 52, 38)]
        # self.ijk_to_track = [
        #             # FM1 TS1
        #             (75, 108, 17), (75, 108, 39), (75, 108, 62),
        #             (75, 109, 2), (75, 109, 24), (75, 109, 47),
        #
        #             # FM1 TS2
        #             (75, 108, 16), (75, 108, 41), (75, 108, 66),
        #             (75, 109, 3), (75, 109, 28), (75, 109, 53),
        #
        #             # FM1 TS3
        #             (75, 108, 17), (75, 108, 37), (75, 108, 56),
        #             (75, 109, 2), (75, 109, 22), (75, 109, 41),
        #
        #             # FM2 TS1
        #             (75, 108, 17), (75, 108, 39), (75, 108, 62),
        #             (75, 109, 2), (75, 109, 24), (75, 109, 47),
        #
        #             # FM2 TS2
        #             (75, 108, 17), (75, 108, 39), (75, 108, 62),
        #             (75, 109, 2), (75, 109, 24), (75, 109, 47),
        #
        #             # FM2 TS3
        #             (75, 108, 17), (75, 108, 36), (75, 108, 56),
        #             (75, 109, 2), (75, 109, 21), (75, 109, 41),
        #
        #             # FM3 TS1
        #             (75, 108, 5), (75, 108, 33), (75, 108, 64),
        #             (75, 109, 2), (75, 109, 30), (75, 109, 61),
        #
        #             # FM3 TS2
        #             (75, 108, 5), (75, 108, 32), (75, 108, 60),
        #             (75, 108, 5), (75, 109, 29), (75, 109, 57),
        #
        #             # FM3 TS3
        #             (75, 108, 5), (75, 108, 32), (75, 108, 56),
        #             (75, 108, 5), (75, 109, 29), (75, 109, 53),
        #         ]
        self.ijk_to_track = []

        self.tracked_cells = []

        for (i, j, k) in self.ijk_to_track:
            global_idx = self.reservoir.discr_mesh.get_global_index(i - 1, j - 1, k - 1)
            local_idx = self.reservoir.discr_mesh.global_to_local[global_idx]
            if local_idx >= 0:
                self.tracked_cells.append((i, j, k, int(local_idx)))  # Make sure it's an int!
                print(f"[Tracking] IJK=({i},{j},{k}) => local ID = {local_idx}")
            else:
                print(f"[Tracking] Skipping inactive cell at IJK=({i},{j},{k})")

        print("[Tracker] Found tracked cells:", self.tracked_cells)
        self.my_tracked_pressures = {
            f'P_I{i}_J{j}_K{k}': [] for (i, j, k, _) in self.tracked_cells
        }
        self.my_average_pressure = []

    def set_wells(self):
        # read perforation data from a file
        self.well_cells = []  # NEW for RHS function
        if hasattr(self.idata, 'schfile'):
            # apply to the reservoir; add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for perf_tuple in wdata.perforations:
                    perf = perf_tuple[1]
                    # adjust to account for added overburden layers
                    perf_ijk_new = (perf.loc_ijk[0], perf.loc_ijk[1], perf.loc_ijk[2] + self.idata.geom.burden_layers)
                    self.well_cells.append(perf_ijk_new)  # NEW, for RHS function
                    self.reservoir.add_perforation(wname,
                                                   cell_index=perf_ijk_new,
                                                   well_index=perf.well_index, well_indexD=perf.well_indexD,
                                                   multi_segment=True, verbose=True)
                    #    multi_segment=True, verbose=True)
        else:
            # add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for k in range(1 + self.idata.geom.burden_layers,
                               self.reservoir.nz + 1 - self.idata.geom.burden_layers):
                    self.reservoir.add_perforation(wname,
                                                   cell_index=(wdata.location.I, wdata.location.J, k),
                                                   well_index=None, multi_segment=True, verbose=True)

    def set_initial_pressure_from_file(self, fname: str):
        # set initial pressure
        p_cpp = value_vector()
        load_single_float_keyword(p_cpp, fname, 'PRESSURE', -1)
        p_file = np.array(p_cpp, copy=False)
        p_mesh = np.array(self.reservoir.mesh.pressure, copy=False)
        try:
            actnum = np.array(self.reservoir.actnum, copy=False)  # CPG Reservoir
        except:
            actnum = self.reservoir.global_data['actnum']  # Struct reservoir
        p_mesh[:self.reservoir.mesh.n_res_blocks * 2] = p_file[actnum > 0]

    def well_is_inj(self, wname: str):  # determine well control by its name
        return "INJ" in wname

    def do_after_step(self):
        # save to grdecl file after each time step
        # self.reservoir.save_grdecl(self.get_arrays_gredcl(ith_step), os.path.join(out_dir, 'res_' + str(ith_step+1)))
        self.physics.engine.report()

        print("[Tracker] Entered do_after_step, time =", self.physics.engine.time_data['time'][-1])
        total_pressure = self.physics.engine.X[::3]
        average_pressure = np.mean(total_pressure)

        self.my_tracked_pressures.setdefault("P_MEAN_Reservoir", []).append(average_pressure)

        pressure = total_pressure  # self.physics.engine.X[self.physics.vars.index('P')]
        nx, ny, nz = map(int, self.reservoir.dims)
        # Gather ΔP across all faults
        from collections import defaultdict

        # Fault-wise grouping
        fault_deltas = defaultdict(list)

        for (i1, j1, k1), (i2, j2, k2) in self.reservoir.fault_connections_ijk:
            idx1 = i1 + nx * j1 + nx * ny * k1
            idx2 = i2 + nx * j2 + nx * ny * k2

            idx1_local = self.reservoir.discr_mesh.global_to_local[idx1]
            idx2_local = self.reservoir.discr_mesh.global_to_local[idx2]

            delta = abs(pressure[idx1_local] - pressure[idx2_local])  # NEW

            # delta = abs(pressure[idx1] - pressure[idx2]) #Original
            fault_name = self.reservoir.fault_connection_to_name.get(((i1, j1, k1), (i2, j2, k2)), "UNNAMED")
            fault_deltas[fault_name].append(delta)

        # Mean ΔP across all faults
        all_deltas = [dp for deltas in fault_deltas.values() for dp in deltas]
        avg_dp_all = np.mean(all_deltas)
        self.my_tracked_pressures.setdefault("DP_FAULT_MEAN", []).append(avg_dp_all)

        # Per-fault ΔP
        for fault_name, deltas in fault_deltas.items():
            key = f"DP_FAULT_{fault_name}"
            self.my_tracked_pressures.setdefault(key, []).append(np.mean(deltas))

        # 1. Max ΔP across all faults
        if all_deltas:
            max_dp_all = np.max(all_deltas)
        else:
            max_dp_all = 0.0
        self.my_tracked_pressures.setdefault("DP_FAULT_MAX", []).append(max_dp_all)

        # 2. Per-fault max ΔP
        for fault_name, deltas in fault_deltas.items():
            if deltas:
                max_dp = np.max(deltas)
            else:
                max_dp = 0.0
            key = f"DP_FAULT_{fault_name}_MAX"
            self.my_tracked_pressures.setdefault(key, []).append(max_dp)

        for (i, j, k, local_idx) in self.tracked_cells:
            key_p = f'P_I{i}_J{j}_K{k}'
            key_t = f'T_I{i}_J{j}_K{k}'

            state_idx = local_idx * self.physics.n_vars
            try:
                p = self.physics.engine.X[state_idx]
                T = self.physics.engine.X[state_idx + 2]

                print(f"[Tracker] {key_p} = {p:.2f} bar, {key_t} = {T:.2f} K")

                self.my_tracked_pressures.setdefault(key_p, []).append(p)
                self.my_tracked_pressures.setdefault(key_t, []).append(T)

            except Exception as e:
                print(f"[Tracker Error] {key_p}/{key_t}: {e}")

        # self.print_well_rate()

    def check_conductivity(self):
        """
        Verifies the conductivity values assigned based on SATNUM.
        Ensures that shale and sandstone have the correct properties.
        """
        # Get SATNUM values again for active cells
        satnum_full = np.array(self.reservoir.satnum)  # Full-grid SATNUM
        global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)

        # Use only active cells
        satnum_active = satnum_full[global_to_local >= 0]

        # Get conductivity values
        conduction_values = np.array(self.reservoir.conduction, copy=False)  # Active-cell conductivity

        # Expected values
        expected_shale = self.idata.rock.conduction_shale
        expected_sandstone = self.idata.rock.conduction_sand

        # Count occurrences
        unique_vals, counts = np.unique(conduction_values, return_counts=True)
        shale_count = np.sum(conduction_values == expected_shale)
        sandstone_count = np.sum(conduction_values == expected_sandstone)

        # Print results
        print(f"\n **Conductivity Check Report** ")
        print(f"Unique conductivity values in the reservoir: {dict(zip(unique_vals, counts))}")
        print(f"Shale assignments (SATNUM=3): {shale_count} cells → Expected: {expected_shale} kJ/m/day/K")
        print(
            f"Sandstone assignments (SATNUM=1 or 2): {sandstone_count} cells → Expected: {expected_sandstone} kJ/m/day/K")

        # Debugging: Check if any unexpected values exist
        unexpected_conductivity = [val for val in unique_vals if val not in [expected_shale, expected_sandstone]]
        if unexpected_conductivity:
            print(
                f"Warning: Found unexpected conductivity values: {unexpected_conductivity}. Check SATNUM processing!")

        print("Conductivity verification complete!\n")


class CustomCPGReservoir(CPG_Reservoir):
    def apply_fault_mult(self, faultfile, cell_m, cell_p, mpfa_tran, ids):
        print("[FaultMult] Building fast lookup map...")
        conn_map = {(cell_m[idx], cell_p[idx]): idx for idx in range(len(ids))}

        reading_editnnc = False
        applied = 0
        skipped = 0
        current_fault_label = "UNNAMED"

        # Store fault connections
        self.fault_connections_ijk = []
        self.fault_connections_flat = []
        self.fault_connections_mult = []
        self.fault_connection_to_name = {}

        nx, ny, nz = map(int, self.dims)
        print("[FaultMult] Reading fault file...")

        with open(faultfile) as f:
            for buff in f:
                line = buff.strip()
                if not line:
                    continue

                # Detect fault name from comment lines
                if line.startswith('--'):
                    if "FAULT" in line.upper():
                        current_fault_label = line.strip("- ").upper()
                    continue

                # Toggle based on keywords
                if line.startswith(('MULTIPLY', 'ENDFLUXNUM', 'REGIONS', 'EQLDIMS', 'BOX', 'ENDBOX', 'COPY')):
                    reading_editnnc = False

                if line.startswith('EDITNNC'):
                    reading_editnnc = True
                    continue

                # Parse EDITNNC connections
                if reading_editnnc:
                    if line.endswith('/'):
                        line = line[:-1].strip()

                    parts = line.split()
                    if len(parts) < 7:
                        continue

                    try:
                        i1, j1, k1 = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
                        i2, j2, k2 = int(parts[3]) - 1, int(parts[4]) - 1, int(parts[5]) - 1
                        mult = float(parts[6])
                    except ValueError:
                        skipped += 1
                        continue

                    # Skip negative indices
                    if any(x < 0 for x in [i1, j1, k1, i2, j2, k2]):
                        skipped += 1
                        continue

                    # Skip indices outside grid
                    if any([
                        i1 >= nx, j1 >= ny, k1 >= nz,
                        i2 >= nx, j2 >= ny, k2 >= nz
                    ]):
                        skipped += 1
                        continue

                    # Local connection lookup
                    m_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i1, j1, k1)]
                    p_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i2, j2, k2)]

                    key = (m_idx, p_idx)
                    if key in conn_map:
                        idx = conn_map[key]
                        mpfa_tran[2 * ids[idx]] *= mult
                        applied += 1

                        # Save fault info
                        self.fault_connections_ijk.append(((i1, j1, k1), (i2, j2, k2)))
                        idx1 = i1 + nx * j1 + nx * ny * k1
                        idx2 = i2 + nx * j2 + nx * ny * k2
                        self.fault_connections_flat.append((idx1, idx2))
                        self.fault_connections_mult.append(mult)
                        self.fault_connection_to_name[((i1, j1, k1), (i2, j2, k2))] = current_fault_label
                    else:
                        skipped += 1

        print(f"[FaultMult] Done. Applied: {applied}, Skipped: {skipped}")


