
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
MAX_SIMULATIONS = 10000

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


class Parameterizable:

    def __init__(self, name):
        self.name = name
        self.param_defs = []
        self.param_vals = {}

    def add_parameter(self, param_def, param_val = None):
        self.param_defs.append(param_def)
        self.param_vals[param_def.short_name] = param_val        

    def add_preset(self, short_name, display_name, preset_param_vals):
        """ preset_dict is a dict of parameter short_name:s and values
        """
        pass

    def choose_preset(self, preset_id = 0):
        pass

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
        time_step = None):
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
            ParameterType.COLOR))

        

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
            'Plot base scenario (without ego driver response)', '', ParameterType.BOOLEAN), True)

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
        if self.param_vals['plot_base_scenario']:
            fig, axs = plt.subplots(4, 1, sharex = True)
            axs[0].plot(self.scenario.time_stamp,  self.scenario.cp_acceleration,
                '-', color = CP_COLOR)
            axs[1].plot(self.scenario.time_stamp, self.scenario.cp_speed, 
                '-', color = CP_COLOR)
            axs[1].plot(self.scenario.time_stamp, self.scenario.ego_speed, 
                '--', color = EGO_COLOR)
            axs[2].plot(self.scenario.time_stamp, self.scenario.distance_gap, 
                '-', color = CP_COLOR)
            axs[3].plot(self.scenario.time_stamp, self.scenario.thetaDot, 
                '-', color = CP_COLOR)
            plt.show()




class FixedLDTModel(DriverModel):

    def __init__(self):
        super().__init__('Looming detection threshold model', (DriverModelCapability.DETECTION_TIME,), 
            is_probabilistic = False, time_step = 0.001)
        self.add_parameter(ParameterDefinition('thetaDot_d', 'Looming detection threshold', 
            'rad/s', ParameterType.FLOAT, (0, MAX_LDT)))

    def simulate_scenario_once(self, i_simulation):
        above_threshold_samples = np.nonzero(self.scenario.thetaDot > 
            self.param_vals['thetaDot_d'])[0]
        if len(above_threshold_samples) == 0:
            detection_time = math.nan
        else:
            detection_time = self.scenario.time_stamp[above_threshold_samples[0]]
        self.outputs[DriverModelCapability.DETECTION_TIME][i_simulation] = detection_time




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

    sim_engine = SimulationEngine(scenario)
    sim_engine.param_vals['end_time'] = 15
    sim_engine.param_vals['n_simulations'] = 100
    sim_engine.add_driver_model(fixed_ldt_model)
    sim_engine.simulate_driver_models()

    print(fixed_ldt_model.outputs)
