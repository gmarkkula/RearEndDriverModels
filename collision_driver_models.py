
import math
from enum import Enum
import numpy as np


MAX_SPEED = 50 # m/s
MAX_TIME = 30 # s
MAX_ACC = 10 # m/s^2
MAX_SIMULATIONS = 10000

class ParameterType(Enum):
    BOOLEAN = 0
    FLOAT = 1
    INTEGER = 2


class ParameterDefinition:
    def __init__(self, short_name, display_name, 
                 param_type = ParameterType.FLOAT, param_unit = None, 
                 param_range = None):
        self.short_name = short_name
        self.display_name = display_name
        self.type = param_type
        self.unit = param_unit
        self.range = param_range


class Parameterizable:

    def __init__(self):
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
        super().__init__()
        self.add_parameter(ParameterDefinition('v_L', 'Initial lead vehicle speed', 
            'm/s', ParameterType.FLOAT, (0, MAX_SPEED)))
        self.add_parameter(ParameterDefinition('T_L,B', 'Lead vehicle acceleration onset time', 
            's', ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('a_L', 'Lead vehicle acceleration magnitude', 
            'm/s^2', ParameterType.FLOAT, (-MAX_ACC, MAX_ACC)))
        self.add_parameter(ParameterDefinition('v_E', 'Initial ego vehicle speed', 
            'm/s', ParameterType.FLOAT, (0, MAX_SPEED)))
        self.add_parameter(ParameterDefinition('T_E,G', 'Ego driver off-road glance end time', 
            's', ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('end_time', 'Simulation end time', 's', 
            ParameterType.FLOAT, (0, MAX_TIME)))
        self.add_parameter(ParameterDefinition('n_simulations', 
            'No. of simulations', '-', ParameterType.INTEGER, (0, MAX_SIMULATIONS)))

    def set_time_series_vectors(self, time_step):
        """ Sets time series vectors for the scenario, with the specified time
            step.
        """

        ### NOTE: Maybe change back to having no of simulations (and also end time?) 
        # as part of a SimulationControl class instead? And pass the end time as another argument here...

        # time stamps
        self.time_step = time_step
        self.time_stamp = np.arange(0, self.param_vals['end_time'], time_step)
        self.n_time_steps = len(self.time_stamp)
        # lead vehicle vectors
        ### TODO
        # ego vehicle vectors
        ### TODO



class DriverModelCapability(Enum):
    DETECTION_TIME = 0
    BRAKE_ONSET_TIME = 1
    BRAKE_CTRL = 2




class DriverModel(Parameterizable):
    
    def __init__(self, capabilities, is_probabilistic = False, time_step = None):
        """ capabilities: tuple of DriverModelCapability
        """
        super().__init__()
        self.capabilities = capabilities
        self.is_probabilistic = is_probabilistic
        if (DriverModelCapability.BRAKE_CTRL in capabilities) and (time_step is None):
            raise Exception('Time step needed if model is doing braking control.')
        self.time_step = time_step

        

    def simulate(self, scenario):
        # how many simulations to run?
        if self.is_probabilistic:
            n_simulations = scenario.param_vals['n_simulations']
        else:
            n_simulations = 1
        # set the time series vectors for the scenario, using the model's time step
        self.scenario = scenario
        self.scenario.set_time_series_vectors(self.time_step)
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



if __name__ == "__main__":

    s = Scenario()
    s.param_vals['v_L'] = 10
    s.param_vals['end_time'] = 10
    s.param_vals['n_simulations'] = 2
    print(s.param_vals)

    capabs = (DriverModelCapability.BRAKE_CTRL, DriverModelCapability.DETECTION_TIME)
    d = DriverModel(capabs, is_probabilistic = True, time_step = 1)
    print(d.capabilities)
    d.simulate(s)
    for output in d.outputs.values():
       print(output)
