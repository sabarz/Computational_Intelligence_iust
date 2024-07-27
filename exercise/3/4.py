import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import gymnasium as gym

position_range = np.linspace(-1.2, 0.6, 100)
velocity_range = np.linspace(-0.07, 0.07, 100)
action_range = np.linspace(-1, 1, 100)

velocity_negative = fuzz.trimf(velocity_range, [-0.07, -0.07, -0.001])
velocity_neutral = fuzz.trimf(velocity_range, [-0.005, 0, 0.005])
velocity_positive = fuzz.trimf(velocity_range, [0.001, 0.07, 0.07])

position_low = fuzz.trapmf(position_range, [-1.2, -1.2, -0.52, -0.45])
position_high = fuzz.trapmf(position_range, [-0.52, -0.45, 0.6, 0.6])

action_left = fuzz.trapmf(action_range, [-1, -1, -0.005, 0])
action_neutral = fuzz.trimf(action_range, [-0.005, 0, 0.005])
action_right = fuzz.trapmf(action_range, [0, 0.005, 1, 1])

position = ctrl.Antecedent(position_range, 'position')
velocity = ctrl.Antecedent(velocity_range, 'velocity')
action = ctrl.Consequent(action_range, 'action')

velocity['negative'] = velocity_negative
velocity['neutral'] = velocity_neutral
velocity['positive'] = velocity_positive

position['low'] = position_low
position['high'] = position_high

action['left'] = action_left
action['neutral'] = action_neutral
action['right'] = action_right

rules = [
    ctrl.Rule(position['high'] & velocity['neutral'], action['neutral']),
    ctrl.Rule(position['high'] & velocity['negative'], action['left']),
    ctrl.Rule(position['high'] & velocity['positive'], action['right']),
    ctrl.Rule(position['low'] & velocity['negative'], action['left']),
    ctrl.Rule(position['low'] & velocity['positive'], action['right']),
    ctrl.Rule(position['low'] & velocity['neutral'], action['neutral']),
]

fuzzy_control_system = ctrl.ControlSystem(rules)
fuzzy_simulator = ctrl.ControlSystemSimulation(fuzzy_control_system)

position.view()
velocity.view()
action.view()
plt.show()

def get_fuzzy_action(pos, vel):
    fuzzy_simulator.input['position'] = pos
    fuzzy_simulator.input['velocity'] = vel
    
    try:
        fuzzy_simulator.compute()
    except ValueError as e:
        return 0 

    return fuzzy_simulator.output['action']

environment = gym.make('MountainCarContinuous-v0', render_mode="human")


state, _ = environment.reset()
done = False
step_count = 0

while not done and step_count < 500:
    pos, vel = state    
    
    action_value = get_fuzzy_action(pos, vel)
    state, reward, done, _, _ = environment.step([action_value])
    environment.render()

    step_count += 1 
    if pos >= 0.45:
        print("Goal reached!")
        done = True

if step_count >= 500:
    print("Failed to reach the goal within 500 steps.")

environment.close()