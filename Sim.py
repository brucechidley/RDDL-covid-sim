import time
import jax
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import Kingston_Info

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
# from pyRDDLGym import ExampleManager

from pyRDDLGym.Core.Policies.Agents import RandomAgent

from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController

#agent = None

def use_random():
    global agent
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions,
                        seed=42)

def use_jax():

    global agent
    planner_args, _, train_args = load_config('jax.cfg')
    planner = JaxRDDLBackpropPlanner(rddl=myEnv.model, **planner_args)
    agent = JaxOfflineController(planner, **train_args)

#use_jax()
#use_random()

#stats = agent.evaluate(myEnv, ground_state=False, episodes=1, verbose=True, render=True)

iterations = 20

all_rewards = []

infectious_counts = []

agent_counts = []

for iter in range (0, iterations):

    Kingston_Info.main()

    base_path = 'rddl'
    ENV = 'covid'

    myEnv = RDDLEnv.RDDLEnv(domain='domain.rddl', instance='problem.rddl')
    MovieGen = MovieGenerator('', ENV, myEnv.horizon)
    myEnv.set_visualizer(None, movie_gen=MovieGen, movie_per_episode=False)
    gif_name = f'{ENV}_chart'

    agent = None

    use_jax()

    vaccinated = []
    masked = []

    susceptible = []
    exposed = []
    infectious = []
    recovered = []

    isolating = []
    hospitalized_regular = []
    hospitalized_ICU = []

    all_mask = []
    student_mask = []
    all_vaccinate = []
    student_vaccinate = []

    time_step = []

    agents_infected = []
    total_agent_count = 0

    agent.reset()
    total_reward = 0
    state = myEnv.reset()

    for step in range(myEnv.horizon):


        policy_input = myEnv.sampler.subs

        vaccinated_count = 0
        masked_count = 0
        
        exposed_count = 0
        infectious_count = 0
        recovered_count = 0

        isolating_count = 0
        hospitalized_regular_count = 0
        hospitalized_ICU_count = 0

        action = agent.sample_action(policy_input)
        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward

        #print(f'step       = {step}')
        #print(f'state      = {state}')
        #print(f'action     = {action}')
        #print(f'next state = {next_state}')
        #print(f'reward     = {reward}')

        for key, value in state.items():
            if key.startswith("vaccinated__") and value == True:
                vaccinated_count += 1
            elif key.startswith("masked__") and value == True:
                masked_count += 1
            elif key == "susceptible_count":
                susceptible.append(value)
            elif key.startswith("exposed__") and value == True:
                exposed_count += 1
            elif key.startswith("infectious__") and value == True:
                infectious_count += 1
                if not (key in agents_infected):
                    agents_infected.append(key)
            elif key.startswith("recovered__") and value == True:
                recovered_count += 1
            elif key.startswith("isolating__") and value == True:
                isolating_count += 1
            elif key.startswith("hospitalized_non_ICU__") and value == True:
                hospitalized_regular_count += 1
            elif key.startswith("hospitalized_ICU__") and value == True:
                hospitalized_ICU_count += 1
            elif key == "vaccine_implemented":
                if value == True:
                    all_vaccinate.append(1)
                else:
                    all_vaccinate.append(0)
            elif key == "vaccine_implemented_students":
                if value == True:
                    student_vaccinate.append(1)
                else:
                    student_vaccinate.append(0)
            elif key == "mask_counter":
                if value == 0:
                    all_mask.append(0)
                else:
                    all_mask.append(1)
            elif key == "mask_counter_students":
                if value == 0:
                    student_mask.append(0)
                else:
                    student_mask.append(1)

        vaccinated.append(vaccinated_count)
        masked.append(masked_count)

        exposed.append(exposed_count)
        infectious.append(infectious_count)
        recovered.append(recovered_count)

        isolating.append(isolating_count)
        hospitalized_regular.append(hospitalized_regular_count)
        hospitalized_ICU.append(hospitalized_ICU_count)

        time_step.append(step)

        state = next_state
        if done:
            break
    print(f'episode ended with reward {total_reward}')

    all_rewards.append(total_reward)

    for key, value in next_state.items():
        if key == "susceptible_count":
            susceptible.append(value)
        elif key.startswith("infectious__"):
            total_agent_count += 1
    
    agent_counts.append(total_agent_count)
    infectious_counts.append(len(agents_infected))

    susceptible.pop(0)

    myEnv.close()

    plt.plot(time_step, susceptible, color='green', label='Susceptible')
    plt.plot(time_step, exposed, color='yellow', label='Exposed')
    plt.plot(time_step, infectious, color='red', label='Infectious')
    plt.plot(time_step, recovered, color='blue', label='Recovered')
    plt.title('SEIR Over Time')
    plt.legend()

    save_name = "Images_rec28_6000/SEIR" + str(iter) + ".png"
    plt.savefig(save_name)
    plt.clf()


    plt.plot(time_step, masked, color='red', label='Masked')
    plt.plot(time_step, vaccinated, color='blue', label='Vaccinated')
    plt.title('Masked and Vaccinated Over Time')
    plt.legend()

    save_name = "Images_rec28_6000/masked_vaccinated" + str(iter) + ".png"
    plt.savefig(save_name)
    plt.clf()

    plt.plot(time_step, isolating, color='blue', label='Isolating')
    plt.plot(time_step, hospitalized_regular, color='orange', label='Hospitalized Non ICU')
    plt.plot(time_step, hospitalized_ICU, color='red', label='Hospitalized ICU')
    plt.title('Hospitalization and Isolation over time')
    plt.legend()

    save_name = "Images_rec28_6000/hospitalization" + str(iter) + ".png"
    plt.savefig(save_name)
    plt.clf()

    figure, axis = plt.subplots(2, 2) 

    axis[0, 0].plot(time_step, all_mask, color='blue', label='all_mask') 
    axis[0, 0].set_title("all_mask action over time") 

    axis[0, 1].plot(time_step, student_mask, color='blue', label='student_mask') 
    axis[0, 1].set_title("student_mask action over time") 

    axis[1, 0].plot(time_step, all_vaccinate, color='blue', label='all_vaccinate') 
    axis[1, 0].set_title("all_vaccinate action over time") 

    axis[1, 1].plot(time_step, student_vaccinate, color='blue', label='student_vaccinate') 
    axis[1, 1].set_title("student_vaccinate action over time") 

    save_name = "Images_rec28_6000/actions" + str(iter) + ".png"
    plt.savefig(save_name)
    plt.clf()

percent_infected = []

for i in range (0, len(agent_counts)):
    percent_infected.append(round(((infectious_counts[i]/agent_counts[i]) * 100), 2))

with open('Images_rec28_6000/results_rec28_6000.txt', 'w') as f:
    f.write ("All rewards: " + str(all_rewards))
    f.write('\n')
    f.write("Average of all rewards: " + str(round((sum(all_rewards)/len(all_rewards)), 2)))
    f.write('\n')
    f.write("Agent Counts: " + str(agent_counts))
    f.write('\n')
    f.write("Average number of agents: " + str(round((sum(agent_counts)/len(agent_counts)), 2)))
    f.write('\n')
    f.write("Infectious counts: " + str(infectious_counts))
    f.write('\n')
    f.write("Average agents infected: " + str(round((sum(infectious_counts)/len(infectious_counts)), 2)))
    f.write('\n')
    f.write("Percent infected on each trial: " + str(percent_infected))
    f.write('\n')
    f.write("Average percent infected: " + str(round((sum(percent_infected)/len(percent_infected)), 2)))