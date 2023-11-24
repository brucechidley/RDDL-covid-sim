import time
import jax
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
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

iterations = 20

trials = 20

folder = "Images_rec14_no_intervention_isolation"

all_rewards = []

infectious_counts = []

agent_counts = []

for iter in range (0, iterations):

    all_rewards_iter = []

    infectious_counts_iter = []

    agent_counts_iter = []

    vaccinated_iter = []
    masked_iter = []

    susceptible_iter = []
    exposed_iter = []
    infectious_iter = []
    recovered_iter = []

    isolating_iter = []
    hospitalized_regular_iter = []
    hospitalized_ICU_iter = []

    all_mask_iter = []
    student_mask_iter = []
    all_vaccinate_iter = []
    student_vaccinate_iter = []

    time_step_iter = []

    agents_infected_iter = []

    seeds_iter = []

    Kingston_Info.main()

    shutil.copy2('problem.rddl', str(folder) + '/Problems/problem' + str(iter) + '.rddl')

    base_path = 'rddl'
    ENV = 'covid'

    myEnv = RDDLEnv.RDDLEnv(domain='domain.rddl', instance='problem.rddl')
    MovieGen = MovieGenerator('', ENV, myEnv.horizon)
    myEnv.set_visualizer(None, movie_gen=MovieGen, movie_per_episode=False)
    gif_name = f'{ENV}_chart'

    agent = None

    os.mkdir(str(folder) + "/iter_" + str(iter))

    use_jax()

    for trial in range (0, trials):

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

        myEnv = RDDLEnv.RDDLEnv(domain='domain.rddl', instance='problem.rddl')
        MovieGen = MovieGenerator('', ENV, myEnv.horizon)
        myEnv.set_visualizer(None, movie_gen=MovieGen, movie_per_episode=False)
        gif_name = f'{ENV}_chart'

        agent.reset()

        total_reward = 0
        
        current_seed = np.random.randint(100000)
        seeds_iter.append(current_seed)
        state = myEnv.reset(seed=current_seed)

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
        all_rewards_iter.append(total_reward)

        for key, value in next_state.items():
            if key == "susceptible_count":
                susceptible.append(value)
            elif key.startswith("infectious__"):
                total_agent_count += 1
        
        agent_counts.append(total_agent_count)
        agent_counts_iter.append(total_agent_count)

        infectious_counts.append(len(agents_infected))
        infectious_counts_iter.append(len(agents_infected))

        susceptible.pop(0)

        myEnv.close()

        vaccinated_iter.append(vaccinated)
        masked_iter.append(masked)

        susceptible_iter.append(susceptible)
        exposed_iter.append(exposed)
        infectious_iter.append(infectious)
        recovered_iter.append(recovered)

        isolating_iter.append(isolating)
        hospitalized_regular_iter.append(hospitalized_regular)
        hospitalized_ICU_iter.append(hospitalized_ICU)

        all_mask_iter.append(all_mask)
        student_mask_iter.append(student_mask)
        all_vaccinate_iter.append(all_vaccinate)
        student_vaccinate_iter.append(student_vaccinate)

        time_step_iter.append(time_step)

        agents_infected_iter.append(agents_infected)

        plt.plot(time_step, susceptible, color='green', label='Susceptible')
        plt.plot(time_step, exposed, color='yellow', label='Exposed')
        plt.plot(time_step, infectious, color='red', label='Infectious')
        plt.plot(time_step, recovered, color='blue', label='Recovered')
        plt.title('SEIR Over Time')
        plt.legend()

        save_name = str(folder) + "/iter_" + str(iter) + "/trial_" + str(trial) + "_SEIR.png"
        plt.savefig(save_name)
        plt.clf()


        plt.plot(time_step, masked, color='red', label='Masked')
        plt.plot(time_step, vaccinated, color='blue', label='Vaccinated')
        plt.title('Masked and Vaccinated Over Time')
        plt.legend()

        save_name = str(folder) + "/iter_" + str(iter) + "/trial_" + str(trial) + "_masked_vaccinated.png"
        plt.savefig(save_name)
        plt.clf()

        plt.plot(time_step, isolating, color='blue', label='Isolating')
        plt.plot(time_step, hospitalized_regular, color='orange', label='Hospitalized Non ICU')
        plt.plot(time_step, hospitalized_ICU, color='red', label='Hospitalized ICU')
        plt.title('Hospitalization and Isolation over time')
        plt.legend()

        save_name = str(folder) + "/iter_" + str(iter) + "/trial_" + str(trial) + "_hospitalization.png"
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

        save_name = str(folder) + "/iter_" + str(iter) + "/trial_" + str(trial) + "_actions.png"
        plt.savefig(save_name)
        plt.clf()

    percent_infected_iter = []

    for i in range (0, len(agent_counts_iter)):
        percent_infected_iter.append(round(((infectious_counts_iter[i]/agent_counts_iter[i]) * 100), 2))

    with open(str(folder) + "/iter_" + str(iter) + "/results_rec2_8000__iter_" + str(iter) + ".txt", 'w') as f:
        f.write ("All rewards: " + str(all_rewards_iter))
        f.write('\n')
        f.write("Average of all rewards: " + str(round((sum(all_rewards_iter)/len(all_rewards_iter)), 2)))
        f.write('\n')
        f.write("Agent Counts: " + str(agent_counts_iter))
        f.write('\n')
        f.write("Average number of agents: " + str(round((sum(agent_counts_iter)/len(agent_counts_iter)), 2)))
        f.write('\n')
        f.write("Infectious counts: " + str(infectious_counts_iter))
        f.write('\n')
        f.write("Average agents infected: " + str(round((sum(infectious_counts_iter)/len(infectious_counts_iter)), 2)))
        f.write('\n')
        f.write("Percent infected on each trial: " + str(percent_infected_iter))
        f.write('\n')
        f.write("Average percent infected: " + str(round((sum(percent_infected_iter)/len(percent_infected_iter)), 2)))

    data_dict = {
        "Vaccinated": str(vaccinated_iter),
        "Masked": str(masked_iter),
        "Susceptible": str(susceptible_iter),
        "Exposed": str(exposed_iter),
        "Infectious": str(infectious_iter),
        "Recovered": str(recovered_iter),
        "Isolating": str(isolating_iter),
        "Hospitalized_regular": str(hospitalized_regular_iter),
        "Hospitalized_ICU": str(hospitalized_ICU_iter),
        "All_mask": str(all_mask_iter),
        "Student_mask": str(student_mask_iter),
        "All_vaccinate": str(all_vaccinate_iter),
        "Student_vaccinate": str(student_vaccinate_iter),
        "Time_step": str(time_step_iter),
        "Agents_Infected": str(agents_infected_iter),
        "All_rewards": str(all_rewards_iter),
        "Agent_counts": str(agent_counts_iter),
        "Infectious_counts": str(infectious_counts_iter),
        "Percent_infected": str(percent_infected_iter),
        "Seeds": str(seeds_iter)
    }

    json_object = json.dumps(data_dict, indent=4)    

    with open(str(folder) + "/iter_" + str(iter) + "/data_iter_" + str(iter) + ".json", "w") as outfile:
        outfile.write(json_object)

"""
percent_infected = []

for i in range (0, len(agent_counts)):
    percent_infected.append(round(((infectious_counts[i]/agent_counts[i]) * 100), 2))

with open("Images_rec14_8000/results_rec14_8000_total.txt", 'w') as f:
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
"""