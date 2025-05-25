import numpy as np

from darts.input.input_data import InputData, WellControl
from darts.models.darts_model import DartsModel
from darts.engines import value_vector

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData

from model_cpg import Model_CPG, fmt

from dataclasses import dataclass, field

from darts.engines import well_control_iface

from scipy.special import erf

from case_base import input_data_base


# region Dataclasses
@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float

    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)


class ModelCCS(Model_CPG):
    def __init__(self):
        self.zero = 1e-10
        super().__init__()

    def set_physics(self):  # NEW
        """Assign physical properties, including relative permeability, based on facies (SATNUM)."""
        # Code used in SPE11b
        """Physical properties"""
        # Fluid components, ions and solid
        self.components = ["CO2", "H2O"]
        phases = ["V", "Aq"]
        comp_data = CompData(self.components, setprops=True)

        temperature = None  # if None, then thermal=True

        # nc = len(components)
        if temperature is None:  # if None, then thermal=True
            thermal = True
            state_spec = Compositional.StateSpecification.PT
        else:
            thermal = False
            state_spec = Compositional.StateSpecification.P

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
        flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                     AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                     AQEoS.CompType.ion: AQEoS.Jager2003
                                                     }))
        pr = flash_params.eos_params["PR"].eos
        aq = flash_params.eos_params["AQ"].eos

        flash_params.eos_order = ["PR", "AQ"]

        # Initialize physics model
        self.physics = Compositional(
            components=self.components,
            phases=phases,
            timer=self.timer,
            n_points=self.idata.obl.n_points,
            min_p=self.idata.obl.min_p,
            max_p=self.idata.obl.max_p,
            min_z=self.idata.obl.min_z,
            max_z=self.idata.obl.max_z,
            min_t=self.idata.obl.min_t,
            max_t=self.idata.obl.max_t,
            state_spec=state_spec,
            cache=self.idata.obl.cache
        )

        self.physics.n_axes_points[0] = 1001  # sets OBL points for pressure, From SPE11b

        facies_rel_perm = {  # based on fluidflower paper
            # # **Wells** ##NEW, values don't mind
            # 0: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300, c2=1.5),  #p_entry=0.025602
            0: Corey(nw=4.0, ng=2.0, swc=0.2, sgc=0.3, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300,
                     c2=1.5),  # p_entry=0.025602
            # **Overbank Sand (Facies 2)**
            1: Corey(nw=3.0, ng=1.5, swc=0.2, sgc=0.2, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300,
                     c2=1.5),  # p_entry=0.038706
            # **Shale (Facies 3)**
            2: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.1, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300,
                     c2=1.5)}  # p_entry=1.935314

        # Assign properties per facies using `add_property_region`
        for i, (region, corey_params) in enumerate(facies_rel_perm.items()):  # NEW

            # Original
            # property_container = AlternativeContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw[:2], #NEW, was property_container = PropertyContainer(phases_name=phases, components_name=components,Mw=comp_data.Mw,
            #                                        temperature=temperature, min_z=self.zero/10)

            property_container = PropertyContainer(phases_name=phases, components_name=self.components,
                                                   Mw=comp_data.Mw[:2],
                                                   temperature=temperature, min_z=self.zero / 10)

            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])

            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw[:2])),
                                                  ('Aq', Garcia2001(self.components)), ])

            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(self.components)), ])

            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)), ('Aq', EoSEnthalpy(eos=aq)), ])

            property_container.conductivity_ev = dict([('V', ConstFunc(8.4)), ('Aq', ConstFunc(170.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),  # interesting
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])  # interesting

            # property_container.capillary_pressure_ev = CapillaryPressure(corey_params)  # interesting

            self.physics.add_property_region(property_container, i)  # interesting

            property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "satA": lambda ii=i: self.physics.property_containers[ii].sat[1], }

            def make_x_accessor(ii, jj, cc):
                return lambda: self.physics.property_containers[ii].x[jj, cc]

            for j, phase_name in enumerate(phases):
                for c, component_name in enumerate(self.components):
                    key = f"x{component_name}" if phase_name == 'Aq' else f"y{component_name}"
                    property_container.output_props[key] = make_x_accessor(i, j, c)

        return

    def get_arrays(self, ith_step):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = list(ev_props)
        props_names = props_names

        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)

        return property_array

    def get_arrays_gredcl(self, ith_step):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = list(ev_props)
        props_names = props_names

        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)

        # Assume these are your values
        specgrid = self.reservoir.dims
        coord = self.reservoir.coord  # Should be a NumPy array of shape (4056,)
        zcorn = self.reservoir.zcorn  # Should be a NumPy array of shape (200000,)

        from collections import OrderedDict

        # Create a new ordered dict, starting with dims, coord, zcorn
        ordered_property_array = OrderedDict()

        ordered_property_array['SPECGRID'] = specgrid
        ordered_property_array['COORD'] = coord
        ordered_property_array['ZCORN'] = zcorn

        # Add the rest of your properties from the original dict
        for key, value in property_array.items():
            ordered_property_array[key] = value

        return ordered_property_array

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='thermal', type_mech='none',
                               init_type='gradient')  # if I want to switch to isothermal? => thermal, if I want to use gradient => gradient

        input_data_base(self.idata, case)
        # set_input_data(self.idata, case)
        self.idata.faultfile = self.idata.faultfile  # NEW

        self.idata.geom.burden_layers = 0

        self.idata.obl.n_points = 1000
        self.idata.obl.zero = 1e-11
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 400.
        self.idata.obl.min_t = 273.15
        self.idata.obl.max_t = 373.15
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        self.idata.obl.cache = False  # True #False
        self.idata.obl.thermal = True  # True #This sets case to non-isothermal #This one doesn't matter

    def set_initial_conditions(self):

        self.temperature_initial_ = 273.15 + 76.85  # K
        self.initial_values = {"pressure": 100.,
                               "H2O": 0.99995,
                               "CO2": 0.00005,
                               "temperature": self.temperature_initial_
                               }

        pressure_grad = 97.75
        temperature_grad = 30
        input_depth = [np.amin(self.reservoir.mesh.depth), np.amax(self.reservoir.mesh.depth)]

        input_distribution = {
            'pressure': [
                1 + input_depth[0] * pressure_grad / 1000,
                1 + input_depth[1] * pressure_grad / 1000
            ],
            'H2O': [
                0.99995, 0.99995
            ],
            'CO2': [
                0.00005, 0.00005
            ],
            'temperature': [
                20 + 273.15 + input_depth[0] * temperature_grad / 1000,
                20 + 273.15 + input_depth[1] * temperature_grad / 1000
            ]
        }
        return self.physics.set_initial_conditions_from_depth_table(
            self.reservoir.mesh,
            input_distribution=input_distribution,
            input_depth=input_depth
        )

    def set_well_controls(self, time: float = 0., verbose=True, type: str = 'bhp'):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        wdata = self.idata.well_data
        wells = wdata.wells  # short name

        i, j, k = 15, 15, 1 #90, 95, 1  # do this manually - not nice but easy
        nx, ny, nz = int(self.reservoir.dims[0]), int(self.reservoir.dims[1]), int(self.reservoir.dims[2])

        res_block_local = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
        well_head_depth = self.reservoir.depth[res_block_local]
        pressure_gradient = 97.75  # bar / km
        well_head_depth_inj_pressure = 1 + well_head_depth * pressure_gradient / 1000 + 10  # dP of 5 bar
        print('well_head_depth_pressure = ', well_head_depth_inj_pressure)
        # well_head_depth_inj_pressure = 160
        mt = 1e9  # kg
        kg_per_kmol = 44.01  # kg/kmol
        inj_rate = mt / kg_per_kmol / 365.25  # kmol/day

        if self.bhp_yes:
            print("The string 'wbhp' is found as control!")
            for w in wells:
                wdata.add_inj_bhp_control(name=w, bhp=well_head_depth_inj_pressure, temperature=300, phase_name='V')
                type = 'bhp'

        else:
            print("The string 'rate' is found as control!")
            for w in wells:
                control_type = well_control_iface.WellControlType.MOLAR_RATE  # kmol/day
                wdata.add_inj_rate_control(name=w, rate=inj_rate, rate_type=control_type, bhp_constraint=250,
                                           temperature=300,
                                           phase_name='V')  # rate=5e5/8 is approximately 1 Mt per year #was rate = 6e6 # kmol/day | bars | K
                type = 'rate'

        inj_stream_base = [1 - self.zero * 100, self.zero * 100]
        eps_time = 1e-15
        for w in self.reservoir.wells:
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:
                continue
            if wctrl.type == 'inj':  # INJ well
                inj_stream = value_vector(inj_stream_base)
                # if wctrl.mode == 'rate' or type == 'rate': # rate control
                if type == 'rate':  # rate control
                    control_type = well_control_iface.WellControlType.MOLAR_RATE  # kmol/day
                    control_type_constraint = well_control_iface.WellControlType.NONE
                    w.set_rate_control(True, control_type, 0, inj_rate, inj_stream, 300)  # 0 is phase_index
                    w.set_rate_constraint(True, control_type_constraint, 0, inj_rate, inj_stream, 300)

                elif type == 'bhp':  # BHP control
                    w.set_bhp_control(True, well_head_depth_inj_pressure, inj_stream, 300)

                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)

            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate':  # rate control
                    w.control = self.physics.new_rate_prod(wctrl.rate, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_prod(wctrl.bhp_constraint)
                elif wctrl.mode == 'bhp':  # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.bhp)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)

        return


class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "Aq":
            self.k_rw_e = corey.krwe
            self.swc = corey.swc
            self.sgc = 0
            self.nw = corey.nw
        else:
            self.k_rg_e = corey.krge
            self.sgc = corey.sgc
            self.swc = 0
            self.ng = corey.ng

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r


class CapillaryPressure:
    def __init__(self, corey):
        self.nph = 2
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        self.eps = 1e-3

    def evaluate(self, sat):
        '''
        default evaluator of capillary pressure Pc based on pow
        :param sat: saturation
        :return: Pc
        '''
        if self.nph > 1:
            Se = (sat[1] - self.swc) / (
                    1 - self.swc)  # Here SAT[1], since first V, then AQ (propertycontainer) #Was Sat[0] for other script, but most likely should be sat[1] here
            if Se < self.eps:
                Se = self.eps
            pc = self.p_entry * Se ** (-1 / self.labda)

            Pc = np.zeros(self.nph, dtype=object)
            Pc[1] = pc  # Shouldn't this be Pc[0]?
        else:
            Pc = [0.0]
        return Pc  # V, Aq


class AlternativeContainer(PropertyContainer):
    def run_flash(self, pressure, temperature, zc):
        # Normalize fluid compositions
        zc_norm = zc if not self.ns else zc[:self.nc_fl] / (1. - np.sum(zc[self.nc_fl:]))

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        try:
            self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1

        # Set present phase idxs
        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0], dtype=int)

        if ph.size == 1:
            self.x[ph[0]] = zc_norm

        return ph

    def evaluate_at_cond_own(self, pressure, temperature, zc):  # state):
        # Composition vector and pressure from state:
        pressure, temperature, zc = pressure, temperature, zc  # self.get_state(state)

        ph = self.run_flash(pressure, temperature, zc)
        self.temperature = self.flash_ev.get_flash_results().temperature

        for j in ph:
            M = np.sum(self.Mw * self.x[j][:])  # molar weight of mixture
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, self.temperature, self.x[j][:]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m

