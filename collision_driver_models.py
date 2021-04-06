
import math
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

MAX_TIME_GAP = 10 # m/s
MIN_WIDTH = 0.1 # m
MAX_WIDTH = 10 # m
MAX_SPEED = 50 # m/s
MAX_TIME = 30 # s
MAX_ACC = 10 # m/s^2
MAX_LDT = 0.05 # rad / s
MAX_INVTAU = 1 # 1/s
MAX_PRT = 3 # s
MAX_SIMULATIONS = 10000
PLOT_TIME_STEP = 0.01 # s

CP_COLOR = 'black'
EGO_COLOR = 'gray'

class ParameterType(Enum):
    BOOLEAN = 0
    FLOAT = 1
    INTEGER = 2
    COLOR = 3

class DriverModelCapability(Enum):
    DETECTION_TIME = 0
    BRAKE_ONSET_TIME = 1
    BRAKE_CTRL = 2


class ParameterDefinition:
    def __init__(self, short_name, display_name, param_unit = None, 
                 param_type = ParameterType.FLOAT, 
                 param_range = None):
        self.short_name = short_name
        self.display_name = display_name
        self.unit = param_unit
        self.type = param_type
        self.range = param_range


class PresetParameters:
    def __init__(self, display_name, preset_param_vals):
        """ preset_param_vals should be a dict with parameter short_names as 
            keys.
        """
        self.display_name = display_name
        self.param_vals = preset_param_vals


class Parameterizable:

    def __init__(self, name):
        self.name = name
        self.param_defs = []
        self.param_vals = {}
        self.presets = []


    def add_parameter(self, param_def, param_val = None):
        self.param_defs.append(param_def)
        self.param_vals[param_def.short_name] = param_val        

    def add_preset(self, preset):
        """ preset_dict is a dict of parameter short_name:s and values.
            NB: it will never be checked whether the preset sets all the
            "important" values of the parameterizable, and it will only be
            checked at choose_preset() time whether all the short_name:s in 
            the provided preset is actually among the Parameterizable's 
            parameters
        """
        self.presets.append(preset)

    def choose_preset(self, preset_id):
        """ preset_id can either be an integer index or a display name string
        """
        # figure out which preset is being requested
        if type(preset_id) is int:
            preset = self.presets[preset_id]
        else:
            found_it = False
            for preset in self.presets:
                if preset.display_name == preset_id:
                    found_it = True
                    break
            if not found_it:
                raise Exception('No preset with display name "' + preset + '".')
        # set parameter values from preset
        for short_name in preset.param_vals.keys():
            self.param_vals[short_name] = preset.param_vals[short_name]

    def active_preset(self):
        pass


class CollisionPartnerTimeSeries:
    def __init__(self, n_time_steps):
        self.acceleration = np.full(n_time_steps, math.nan)
        self.speed = np.full(n_time_steps, math.nan)
        self.rear_pos = np.full(n_time_steps, math.nan)


class EgoVehicleTimeSeries:
    def __init__(self, n_time_steps, n_simulations):
        if n_simulations == 1:
            nan_array = np.full(n_time_steps, math.nan)
        else:
            nan_array = np.full((n_simulations, n_time_steps))
        self.acceleration = np.copy(nan_array)
        self.speed = np.copy(nan_array)
        self.front_pos = np.copy(nan_array)
        self.eyes_on_road = np.copy(nan_array)
        self.distance_gap = np.copy(nan_array)
        self.speed_diff = np.copy(nan_array)
        self.theta = np.copy(nan_array)
        self.thetaDot = np.copy(nan_array)
        self.invTau = np.copy(nan_array)
    def set_looming_arrays(self, cp_width):
        """ Assumes the members distance_gap and speed_diff have already been
            set, and calculates theta, thetaDot and invTau from these
        """
        self.theta = 2 * np.arctan(cp_width / self.distance_gap)
        self.thetaDot = -cp_width * self.speed_diff / (
            self.distance_gap ** 2 + cp_width ** 2 / 4)
        self.invTau = self.thetaDot / self.theta


class Scenario(Parameterizable):

    def __init__(self):
        super().__init__('Scenario')
        self.add_parameter(ParameterDefinition('v_E', 'Initial speed of ego vehicle', 
            'm/s', ParameterType.FLOAT, (0, MAX_SPEED)))
        self.add_parameter(ParameterDefinition('T_o', 'End time of ego driver off-road glance', 
            's', ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('T_g', 'Initial time gap between ego vehicle and collision partner', 
            's', ParameterType.FLOAT, (0, MAX_TIME_GAP)))
        self.add_parameter(ParameterDefinition('W_P', 'Width of collision partner', 
            'm', ParameterType.FLOAT, (MIN_WIDTH, MAX_WIDTH)))
        self.add_parameter(ParameterDefinition('v_P', 'Initial speed of collision partner', 
            'm/s', ParameterType.FLOAT, (0, MAX_SPEED)))
        self.add_parameter(ParameterDefinition('T_a', 'Onset time of collision partner acceleration', 
            's', ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('a_P', 'Acceleration magnitude of collision partner', 
            'm/s²', ParameterType.FLOAT, (-MAX_ACC, MAX_ACC)))



    def set_time_series_arrays(self, time_step, end_time):
        """ Sets time series arrays for the scenario, with the specified time
            step.
        """
        # time stamps
        self.time_step = time_step
        self.end_time = end_time
        self.time_stamp = np.arange(0, end_time, time_step)
        self.n_time_steps = len(self.time_stamp)       
        # collision partner arrays
        self.cp = CollisionPartnerTimeSeries(self.n_time_steps) 
        self.cp.acceleration = np.zeros(self.n_time_steps)
        self.cp.acceleration[self.time_stamp >= 
            self.param_vals['T_a']] = self.param_vals['a_P']
        self.cp.speed = np.maximum(0, self.param_vals['v_P'] + np.cumsum(
            self.cp.acceleration * time_step))
        cp_rear_pos0 = self.param_vals['T_g'] * self.param_vals['v_E']
        self.cp.rear_pos = cp_rear_pos0 + np.cumsum(self.cp.speed * time_step)
        # ego vehicle arrays
        self.ego = EgoVehicleTimeSeries(self.n_time_steps, 1)
        self.ego.front_pos = self.time_stamp * self.param_vals['v_E']
        self.ego.speed = np.full(self.n_time_steps, self.param_vals['v_E'])
        self.ego.eyes_on_road = self.time_stamp >= self.param_vals['T_o']
        self.ego.distance_gap = self.cp.rear_pos - self.ego.front_pos
        self.ego.speed_diff = self.cp.speed - self.ego.speed
        self.ego.set_looming_arrays(self.param_vals['W_P'])
        # add info on first eyes on road sample
        ego_eyes_on_road_samples = np.nonzero(self.ego.eyes_on_road)[0]
        if len(ego_eyes_on_road_samples):
            self.ego.eyes_on_road_sample = math.nan
        else:
            self.ego.eyes_on_road_sample = ego_eyes_on_road_samples[0]




class DriverModel(Parameterizable):
    
    def __init__(self, name, capabilities, is_probabilistic = False, 
        time_step = None, plot_color = 'black'):
        """ capabilities: tuple of DriverModelCapability
        """
        super().__init__(name)
        self.capabilities = capabilities
        self.is_probabilistic = is_probabilistic
        if (DriverModelCapability.BRAKE_CTRL in capabilities) and (
            time_step is None):
            raise Exception('Time step needed if model is doing braking control.')
        self.time_step = time_step
        self.add_parameter(ParameterDefinition('color', 'Plot color', None, 
            ParameterType.COLOR), plot_color)

        

    def simulate(self, scenario, end_time, n_simulations):
        # set the time series vectors for the scenario, using the model's time step
        self.scenario = scenario
        self.scenario.set_time_series_arrays(self.time_step, end_time)
        # prepare the dictionary of model outputs
        self.outputs = {}
        for capab in self.capabilities:
            if capab == DriverModelCapability.BRAKE_CTRL:
                self.outputs[capab] = np.full(
                    (n_simulations, self.scenario.n_time_steps), math.nan)
            else:
                self.outputs[capab] = np.full(n_simulations, math.nan)
        # do any other preparations needed before running individual simulations
        self.prepare_simulation()
        # run the simulations
        for i in range(n_simulations):
            self.simulate_scenario_once(i)

    def prepare_simulation(self):
        """ To be overridden if needed.
        """
        pass

    def simulate_scenario_once(self, i_simulation):
        """ To be overridden. Should write to the appropriate place in the
            self.outputs[] vectors:
            * 
        """
        pass




class SimulationEngine(Parameterizable):

    def __init__(self, scenario):
        super().__init__('Simulation/plotting')
        self.add_parameter(ParameterDefinition('end_time', 'Simulation/plotting end time', 's', 
            ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('n_simulations', 
            'No. of simulations (probabilistic models only)', '-', ParameterType.INTEGER, (0, MAX_SIMULATIONS)))
        self.add_parameter(ParameterDefinition('plot_base_scenario', 
            'Plot base scenario', '', ParameterType.BOOLEAN), True)

        self.scenario = scenario
        self.clear_driver_models()

    def clear_driver_models(self):
        self.driver_models = []

    def add_driver_model(self, model):
        self.driver_models.append(model)

    def simulate_driver_models(self):
        for model in self.driver_models:
            if model.is_probabilistic:
                n_simulations = self.param_vals['n_simulations']
            else:
                n_simulations = 1
            model.simulate(self.scenario, self.param_vals['end_time'], n_simulations)

    def plot(self):
        self.scenario.set_time_series_arrays(PLOT_TIME_STEP, 
            self.param_vals['end_time'])
        # base scenario plotting
        if self.param_vals['plot_base_scenario']:
            fig, axs = plt.subplots(6, 1, sharex = True, figsize = (8, 8))
            axs[0].plot(self.scenario.time_stamp,  self.scenario.cp.acceleration,
                '--', color = CP_COLOR)
            axs[0].set_ylabel('Acceleration (m/s$^2$)')
            axs[1].plot(self.scenario.time_stamp, self.scenario.ego.speed, 
                ':', color = EGO_COLOR)
            axs[1].plot(self.scenario.time_stamp, self.scenario.cp.speed, 
                '--', color = CP_COLOR)
            axs[1].set_ylabel('Speed (m/s)')
            axs[1].legend(('ego vehicle', 'coll. partner'))
            axs[2].plot(self.scenario.time_stamp, self.scenario.ego.distance_gap, 
                '-', color = CP_COLOR)
            axs[2].set_ylabel('Distance gap (m)')
            axs[3].plot(self.scenario.time_stamp, self.scenario.ego.thetaDot, 
                '-', color = CP_COLOR)
            axs[3].set_ylabel(r'$d\theta/dt$ (rad/s)')
            axs[3].set_ylim((0, MAX_LDT))
            axs[4].plot(self.scenario.time_stamp, self.scenario.ego.invTau, 
                '-', color = CP_COLOR)
            axs[4].set_ylabel(r'$\tau^{-1}$ (s$^{-1}$)')
            axs[4].set_ylim((0, MAX_INVTAU))
            axs[5].plot(self.scenario.time_stamp, self.scenario.ego.eyes_on_road, 
                ':', color = EGO_COLOR)
            axs[5].set_ylabel('Eyes on road (-)')
            axs[5].set_xlabel('Time (s)')
        # figure out what types of model capabilities to plot
        plot_capabs = {}
        for capab in DriverModelCapability:
            plot_capabs[capab] = False
        for driver_model in self.driver_models:
            for capab in driver_model.capabilities:
                plot_capabs[capab] = True
        # plot detection time and brake onset time
        for time_capab in (DriverModelCapability.DETECTION_TIME, 
            DriverModelCapability.BRAKE_ONSET_TIME):
            if plot_capabs[time_capab]:
                fig, ax = plt.subplots(figsize = (8, 2))
                for driver_model in self.driver_models:
                    if time_capab in driver_model.capabilities:
                        if not driver_model.is_probabilistic:
                            ax.axvline(driver_model.outputs[time_capab], 
                                color = driver_model.param_vals['color'])
                            ax.set_xlim((0, self.param_vals['end_time']))
                if time_capab == DriverModelCapability.DETECTION_TIME:
                    ax.set_xlabel('Detection time (s)')
                else:
                    ax.set_xlabel('Brake onset time (s)')
                fig.set_tight_layout(True)

        # show the plots
        plt.show()
        
        




class FixedLDTModel(DriverModel):

    def __init__(self):
        super().__init__('Looming detection threshold model', (DriverModelCapability.DETECTION_TIME,), 
            is_probabilistic = False, time_step = 0.001, plot_color = 'magenta')
        self.add_parameter(ParameterDefinition('thetaDot_d', 'Looming detection threshold', 
            'rad/s', ParameterType.FLOAT, (0, MAX_LDT)))
        self.add_preset(PresetParameters('Hoffman and Mortimer (1994)', {'thetaDot_d': 0.003}))
        self.add_preset(PresetParameters('Markkula et al. (2020)', {'thetaDot_d': 0.002}))
        self.choose_preset('Hoffman and Mortimer (1994)')

    def simulate_scenario_once(self, i_simulation):
        above_threshold_samples = np.nonzero(self.scenario.ego.thetaDot > 
            self.param_vals['thetaDot_d'])[0]
        if len(above_threshold_samples) == 0:
            detection_time = math.nan
        else:
            detection_time = self.scenario.time_stamp[above_threshold_samples[0]]
        self.outputs[DriverModelCapability.DETECTION_TIME][i_simulation] = detection_time



class MaddoxAndKiefer2012Model(DriverModel):

    def __init__(self):
        super().__init__('Maddox and Kiefer (2012)', 
            (DriverModelCapability.DETECTION_TIME, 
            DriverModelCapability.BRAKE_ONSET_TIME), 
            is_probabilistic = False, time_step = 0.001, plot_color = 'red')
        self.add_parameter(ParameterDefinition('thetaDot_d', 'Looming detection threshold', 
            'rad/s', ParameterType.FLOAT, (0, MAX_LDT)))
        self.add_parameter(ParameterDefinition('PRT', 'Perception-reaction time', 
            's', ParameterType.FLOAT, (0, MAX_PRT)))
        self.add_preset(PresetParameters('Low PRT', {'thetaDot_d': 0.0397, 'PRT': 0.75}))
        self.add_preset(PresetParameters('Mid PRT', {'thetaDot_d': 0.0174, 'PRT': 1.5}))
        self.add_preset(PresetParameters('High PRT', {'thetaDot_d': 0.0117, 'PRT': 2}))
        self.choose_preset('Mid PRT')

    def simulate_scenario_once(self, i_simulation):
        above_threshold_samples = np.nonzero(self.scenario.ego.thetaDot > 
            self.param_vals['thetaDot_d'])[0]
        if len(above_threshold_samples) == 0:
            detection_time = math.nan
        else:
            detection_time = self.scenario.time_stamp[above_threshold_samples[0]]
        self.outputs[DriverModelCapability.DETECTION_TIME][i_simulation] = detection_time
        self.outputs[DriverModelCapability.BRAKE_ONSET_TIME][i_simulation] = (
            detection_time + self.param_vals['PRT'])



class MarkkulaEtAl2016Model(DriverModel):

    def __init__(self):
        super().__init__('Markkula et al. (2016)', 
            (DriverModelCapability.BRAKE_ONSET_TIME, 
            DriverModelCapability.BRAKE_CTRL), 
            is_probabilistic = True, time_step = 0.01, plot_color = 'orange')
        self.add_parameter(ParameterDefinition('mu_alpha_on', r'$\alpha_B$ eyes on threat: $\mu$', 
            '-', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('sigma_alpha_on', r'$\alpha_B$ eyes on threat: $\sigma$', 
            '-', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('mu_alpha_off', r'$\alpha_B$ eyes off threat: $\mu$', 
            '-', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('sigma_alpha_off', r'$\alpha_B$ eyes off threat: $\sigma$', 
            '-', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('mu_k', r'$k_B$: $\mu$', 
            'g', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('sigma_k', r'$k_B$: $\sigma$', 
            'g', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('mu_a', r'$a_1$: $\mu$', 
            'm/s²', ParameterType.FLOAT, (-10, 10)))
        self.add_parameter(ParameterDefinition('sigma_a', r'$a_1$: $\sigma$', 
            'm/s²', ParameterType.FLOAT, (-10, 10)))
        self.add_preset(PresetParameters('SHRP 2 car near-crashes/crashes', 
            {'mu_alpha_on': 0.22, 'sigma_alpha_on': 0.29, 
            'mu_alpha_off': -1.40, 'sigma_alpha_off': 0.71,
            'mu_k': 1.23, 'sigma_k': 0.62, 'mu_a': -6.68, 'sigma_a': 1.69}))
        self.add_preset(PresetParameters('SHRP 2 truck/bus near-crashes/crashes', 
            {'mu_alpha_on': 0.27, 'sigma_alpha_on': 0.40, 
            'mu_alpha_off': -0.51, 'sigma_alpha_off': 0.85,
            'mu_k': 0.84, 'sigma_k': 1.44, 'mu_a': -4.70, 'sigma_a': 1.44}))
        self.choose_preset('SHRP 2 car near-crashes/crashes')

    def prepare_simulation(self):
        # determine if the scenario is eyes-on-threat or eyes-off-threat
        pass
        # find time t_C of non-response collision

    def simulate_scenario_once(self, i_simulation):
        # get alpha_B
        pass
        # get time of brake onset t_B

        # get brake ramp-up jerk j_B

        # get maximum deceleration a_1

        # construct the acceleration signal

        # figure out the impactof this acceleration signal



if __name__ == "__main__":

    scenario = Scenario()
    scenario.param_vals['v_E'] = 10
    scenario.param_vals['T_g'] = 4
    scenario.param_vals['T_o'] = 0
    scenario.param_vals['W_P'] = 1.9
    scenario.param_vals['v_P'] = 10
    scenario.param_vals['T_a'] = 0
    scenario.param_vals['a_P'] = -0.35

    fixed_ldt_model = FixedLDTModel()
    fixed_ldt_model.param_vals['thetaDot_d'] = 0.002

    mk2012_model = MaddoxAndKiefer2012Model()

    sim_engine = SimulationEngine(scenario)
    sim_engine.param_vals['end_time'] = 15
    sim_engine.param_vals['n_simulations'] = 100
    sim_engine.add_driver_model(fixed_ldt_model)
    sim_engine.add_driver_model(mk2012_model)
    sim_engine.simulate_driver_models()
    sim_engine.plot()
