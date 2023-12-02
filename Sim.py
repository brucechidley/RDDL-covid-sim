#Author: Bruce Chidley
#This file runs simulations for the COVID-19 RDDL model
#Will call upon Kingston_Info.py, so ensure that the arguments are defined as needed

import time
import jax
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
from pprint import pprint
from pathlib import Path

import Kingston_Info

import argparse

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator

from pyRDDLGym.Core.Policies.Agents import RandomAgent

from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController

#Defining the two types of planners that can be used. When simulating without actions, use_random() is much faster. When using actions, must do use_jax()
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

def parse_arguments():

    parser = argparse.ArgumentParser(description="Configure simulation parameters")

    #Residence populations. The number of students in residence at each post-secondary institution (default ratio roughly corresponds to real-life numbers)
    parser.add_argument("--queens_residence_pop", type=int, default=30, help="Enter the Queen's residence population")
    parser.add_argument("--slc_residence_pop", type=int, default=3, help="Enter the SLC residence population")
    parser.add_argument("--rmc_residence_pop", type=int, default=10, help="Enter the RMC residence population")

    #Total post-secondary school populations
    parser.add_argument("--queens_pop", type=int, default=40, help="Enter the total Queen's population")
    parser.add_argument("--slc_pop", type=int, default=5, help="Enter the total SLC population")
    parser.add_argument("--rmc_pop", type=int, default=13, help="Enter the total RMC population")

    #Kingston population. Roughly the total number of agents that will be in the simulation (can be up to 4 more due to home population generation)
    parser.add_argument("--kingston_pop", type=int, default=100, help="Enter the total population")

    #Number of residences at each post-secondary institution (except for SLC, which only has 1 residence)
    parser.add_argument("--queens_residences", type=int, default=4, help="Enter the number of Queen's residences")
    parser.add_argument("--rmc_residences", type=int, default=1, help="Enter the number of RMC residences")

    #Penalties associated with the simulation and planner actions
    parser.add_argument("--mask_penalty_all", type=float, default=-10, help="Enter the mask penalty factor for all agents")
    parser.add_argument("--vaccine_penalty_all", type=float, default=-10, help="Enter the vaccine penalty factor for all agents")
    parser.add_argument("--mask_penalty_students", type=float, default=-5, help="Enter the mask penalty factor for students")
    parser.add_argument("--vaccine_penalty_students", type=float, default=-5, help="Enter the vaccine penalty factor for students")
    parser.add_argument("--non_icu_penalty", type=float, default=-8000, help="Enter the non-ICU penalty factor")
    parser.add_argument("--icu_penalty", type=float, default=-8000, help="Enter the ICU penalty factor")

    #The factors that will be multiplied with transmission chance
    parser.add_argument("--mask_factor", type=float, default=0.8, help="Enter the factor that wearing a mask multiplies transmission rate by")
    parser.add_argument("--vaccine_factor", type=float, default=0.4, help="Enter the factor that being vaccinated multiplies transmission rate by")

    #The chance an agent wears a mask
    parser.add_argument("--mask_chance", type=float, default=0.7, help="Enter the chance that an agent wears a mask")

    #The total number of non-ICU and ICU beds
    parser.add_argument("--non_icu_beds", type=int, default=2, help="Enter the total number of non-ICU beds")
    parser.add_argument("--icu_beds", type=int, default=1, help="Enter the total number of ICU beds")

    #The number of time steps for the simulation
    parser.add_argument("--horizon", type=int, default=100, help="Enter the desired number of time steps (horizon)")
    
    #Defines the way the simulation is run
    parser.add_argument("--mode", type=str, default="Init", help="Enter the desired mode (Init if you are creating new problem files, Test if you are drawing from existing problem files)")
    parser.add_argument("--iters", type=int, default=20, help="Enter the number of iterations you wish to run")
    parser.add_argument("--trials", type=int, default=20, help="Enter the number of trials per iteration you wish to run")

    return parser.parse_args()

args_sim = parse_arguments()

#Each iteration is a different problem file
iterations = args_sim.iters

#Trials = Number of simulations per problem file 
trials = args_sim.trials

#Folders that the results of the simulation will be saved to (CHANGE AS NEEDED)

folder_list = ["rec2_no_intervention_no_iso"]

#folder_list = ["rec2_iso_03", "rec2_iso_05", "rec2_iso_07"] 

#Domains that the problem files will be run on (CHANGE AS NEEDED)
#Must be the same length as folder_list, and the domains at each index must be for the folders at the same index

domain_list = ["domain_rec2_no_intervention_no_iso"]

#domain_list = ["domain_rec2_iso_03", "domain_rec2_iso_05", "domain_rec2_iso_07"]

#Folder that the problem files will be drawn from (CHANGE AS NEEDED)
#If we are initializing problem files, then we call from folder_list as defined earlier
#If we are drawing from existing problem files, then that should not change for the whole run, and so it is only one folder
if args_sim.mode == "Test":
    #Source file when testing (CHANGE AS NEEDED)
    source_file_folder = "rec2_no_intervention_no_iso"


#Loops over all folders, indicating a different domain/configuration
for folder_num in range (0, len(folder_list)):

    #Tracks the current target and source folder
    folder = folder_list[folder_num]
    Path(folder_list[folder_num]).mkdir(parents=True, exist_ok=True)

    if args_sim.mode == "Init":
        Path(folder_list[folder_num] + "/Problems").mkdir(parents=True, exist_ok=True)
        source_file_folder = folder_list[folder_num]

    #Loops over the number of specified iterations
    for iter in range (0, iterations):

        #Holds final rewards for each trial
        all_rewards_iter = []

        #Holds the number of unique agents who become infectious on each trial
        infectious_counts_iter = []

        #Holds the number of agents per trial
        agent_counts_iter = []
    
        #Holds the number of agents who are vaccinated/masked at each time step on each trial
        vaccinated_iter = []
        masked_iter = []

        #Holds the number of agents in each class at each time step on each trial
        susceptible_iter = []
        exposed_iter = []
        infectious_iter = []
        recovered_iter = []

        #Holds the number of agents isolated/hospitalized at each time step on each trial
        isolating_iter = []
        hospitalized_regular_iter = []
        hospitalized_ICU_iter = []

        #Holds a 1 or 0 depending on whether the action is active or not (1 is active, 0 is not active)
        all_mask_iter = []
        student_mask_iter = []
        all_vaccinate_iter = []
        student_vaccinate_iter = []

        #Tracks the current time step for each trial
        time_step_iter = []

        #Tracks the agents that are infectious on each trial
        agents_infected_iter = []

        #Tracks the seeds for all trials, for future testing purposes
        seeds_iter = []


        #New problem files are created if necessary
        if args_sim.mode == "Init":
            Kingston_Info.main()
            shutil.copy2('problem.rddl', str(folder) + '/Problems/problem' + str(iter) + '.rddl')


        base_path = 'rddl'
        ENV = 'covid'
        
        #Generate the environment based on domain + problem file
        myEnv = RDDLEnv.RDDLEnv(domain='Domains/' + str(domain_list[folder_num]) + '.rddl', instance=str(source_file_folder) + '/Problems/problem' + str(iter) + '.rddl')
        MovieGen = MovieGenerator('', ENV, myEnv.horizon)
        myEnv.set_visualizer(None, movie_gen=MovieGen, movie_per_episode=False)
        gif_name = f'{ENV}_chart'

        agent = None

        #Create iteration folder
        Path(str(folder) + "/iter_" + str(iter)).mkdir(parents=True, exist_ok=True)

        #Create the agent (either jax planner or random)
        use_jax()
        #use_random()

        #Open source json file for the purpose of extracting seeds for each simulation
        #By using the same seed for each trial, per iteration, we can better analyze the effects of isolation and intervention
        if args_sim.mode == "Test":
            f = open(str(source_file_folder) + '/iter_' + str(iter) + '/data_iter_' + str(iter) + '.json')
            data = json.load(f)
            seed_list = eval(data['Seeds'])

        #Loops over all trials
        for trial in range (0, trials):

            #Tracks vaccinated/masked agents per time step
            vaccinated = []
            masked = []

            #Tracks classes for agents per time step
            susceptible = []
            exposed = []
            infectious = []
            recovered = []

            #Tracks isolated/hospitalized for agents per time step
            isolating = []
            hospitalized_regular = []
            hospitalized_ICU = []

            #Tracks actions for agents per time step
            all_mask = []
            student_mask = []
            all_vaccinate = []
            student_vaccinate = []

            #Tracks time steps
            time_step = []

            #Tracks the agents who are infected per time step
            agents_infected = []
            total_agent_count = 0

            #Resets the agent for each trial
            agent.reset()

            #Sets the total reward back to 0 for each trial
            total_reward = 0
        
            #Seed for the trial is either random or pulled from a list of seeds based on the source folder

            if args_sim.mode == "Init":
                current_seed = np.random.randint(100000)
            elif args_sim.mode == "Test":
                current_seed = int(seed_list[trial])

            seeds_iter.append(current_seed)

            #Reset environment based on seed
            state = myEnv.reset(seed=current_seed)

            #Steps through the simulation
            for step in range(myEnv.horizon):
                
                #This is necessary when using JaxPlanner. policy_input goes in the brackets for agent.sample_action()
                #Otherwise, comment this out and put "state" in the brackets for agent.sample_action()
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

                #Loops through all elements in the state dictionary
                for key, value in state.items():

                    #If an agent exhibits the following traits, increase the respective counter by 1
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

                    #If action has been implemented, add 1. If not, add 0
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

            all_rewards_iter.append(total_reward)

            #Susceptible count lags behind by 1, so we get the value of the next step and pop the value of the first step
            for key, value in next_state.items():
                if key == "susceptible_count":
                    susceptible.append(value)
                elif key.startswith("infectious__"):
                    total_agent_count += 1
        
            agent_counts_iter.append(total_agent_count)

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

            #Plot all the graphs
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

        #Calculate the percent infected on each trial
        percent_infected_iter = []

        for i in range (0, len(agent_counts_iter)):
            percent_infected_iter.append(round(((infectious_counts_iter[i]/agent_counts_iter[i]) * 100), 2))

        #Write info to a text file
        with open(str(folder) + "/iter_" + str(iter) + "/"  + str(folder) + "_" + str(iter) + ".txt", 'w') as f:
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

        #Write all detailed trial info to a json file
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