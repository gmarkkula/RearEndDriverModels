
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
            'm/s^2', ParameterType.FLOAT, (-MAX_ACC, MAX_ACC)))



    def set_time_series_arrays(self, time_step, end_time):
        """ Sets time series arrays for the scenario, with the specified time
            step.
        """
        # time stamps
        self.time_step = time_step
        self.end_time = end_time
        self.time_stamp = np.arange(0, end_time, time_step)
        self.n_time_steps = len(self.time_stamp)
        # ego vehicle arrays
        self.ego_front_pos = self.time_stamp * self.param_vals['v_E']
        self.ego_speed = np.full(self.n_time_steps, self.param_vals['v_E'])
        self.ego_eyes_on_road = self.time_stamp >= self.param_vals['T_o']
        # collision partner arrays
        self.cp_acceleration = np.zeros(self.n_time_steps)
        self.cp_acceleration[self.time_stamp >= 
            self.param_vals['T_a']] = self.param_vals['a_P']
        self.cp_speed = np.maximum(0, self.param_vals['v_P'] + np.cumsum(
            self.cp_acceleration * time_step))
        cp_rear_pos0 = self.param_vals['T_g'] * self.param_vals['v_E']
        self.cp_rear_pos = cp_rear_pos0 + np.cumsum(self.cp_speed * time_step)
        self.distance_gap = self.cp_rear_pos - self.ego_front_pos
        self.speed_diff = self.cp_speed - self.ego_speed
        # looming arrays
        self.theta = 2 * np.arctan(self.param_vals['W_P'] / self.distance_gap)
        self.thetaDot = -self.param_vals['W_P'] * self.speed_diff / (
            self.distance_gap ** 2 + self.param_vals['W_P'] ** 2 / 4)




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
        # run the simulations
        for i in range(n_simulations):
            self.simulate_scenario_once(i)


    def simulate_scenario_once(self, i_simulation):
        """ To be overridden. Should write to the appropriate place in the
            self.outputs[] vectors
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
            fig, axs = plt.subplots(4, 1, sharex = True, figsize = (8, 6))
            axs[0].plot(self.scenario.time_stamp,  self.scenario.cp_acceleration,
                '--', color = CP_COLOR)
            axs[0].set_ylabel('Acceleration (m/s$^2$)')
            axs[1].plot(self.scenario.time_stamp, self.scenario.ego_speed, 
                ':', color = EGO_COLOR)
            axs[1].plot(self.scenario.time_stamp, self.scenario.cp_speed, 
                '--', color = CP_COLOR)
            axs[1].set_ylabel('Speed (m/s)')
            axs[1].legend(('ego vehicle', 'coll. partner'))
            axs[2].plot(self.scenario.time_stamp, self.scenario.distance_gap, 
                '-', color = CP_COLOR)
            axs[2].set_ylabel('Distance gap (m)')
            axs[3].plot(self.scenario.time_stamp, self.scenario.thetaDot, 
                '-', color = CP_COLOR)
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel(r'$d\theta/dt$ (rad/s)')
            axs[3].set_ylim((0, MAX_LDT))
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
        above_threshold_samples = np.nonzero(self.scenario.thetaDot > 
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
        above_threshold_samples = np.nonzero(self.scenario.thetaDot > 
            self.param_vals['thetaDot_d'])[0]
        if len(above_threshold_samples) == 0:
            detection_time = math.nan
        else:
            detection_time = self.scenario.time_stamp[above_threshold_samples[0]]
        self.outputs[DriverModelCapability.DETECTION_TIME][i_simulation] = detection_time
        self.outputs[DriverModelCapability.BRAKE_ONSET_TIME][i_simulation] = (
            detection_time + self.param_vals['PRT'])



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
