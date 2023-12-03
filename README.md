# Modeling Disease Spread in a Community

## Overview

This RDDL domain models the spread of disease in a community. Agents, in this domain, move through the following SEIR states:

 - Susceptible: An agent does not have the disease, and has no natural immunity to it
 - Exposed: An agent has contracted the disease, but is asymptomatic and unable to spread the disease to other agents
 - Infectious: An agent is showing symptoms for the disease, and is able to spread it to other agents
 - Recovered: An agent is recovered from the disease, and is unable to contract it

Before describing disease dynamics in detail, I will explain how the agents are initialized and how they move around. Agents have the following properties associated with them:

 - Age Bracket
 - Home (where an agent goes every night)
	 - Only 1-5 agents per home
 - Job (where an agent goes during the day on weekdays)
 - Store (where an agent goes during the day on weekends)

Each day is broken up into two time steps - day and night. On the "day" time step, all agents go to their jobs, and on the "night" time step, all agents go to their homes (these can change when isolating or hospitalized - more on this later). This is tracked by a simple counter, ticking between 0 and 1. Similarly, a counter is used to track weekdays and weekends. Weekdays are represented by counter values 1-10, and weekends are represented by counter values 11-14. Note that traveling between points is not captured here - the agents simply teleport around.

Agents, by default, begin in the susceptible state. When an agent is in the susceptible state, they move around the world freely. When they are at a location, they contract the disease at a probability of (R0/total length of infectious period)/(number of susceptible agents at the location). If they do contract the disease, then they are moved into the exposed class, where they remain for a total of Normal(9,2) time steps (about 4.5 days). From here, they move into the infectious class. When they move into the infectious class, they have a probability of isolating at home (equal to the probability of them wearing a mask). If they choose to isolate, they do not go to their job, and simply remain at home (they can still infect other members of their household). Their infection can also be mild, severe, or critical. In the mild case, it is business as usual. In the severe and critical cases, they become hospitalized. If the infection is severe, then they occupy a regular hospital bed. If the infection is critical, then they occupy an ICU bed. Both of these have limits, and exceeding these limits results in a penalty (more on this later). If an agent is hospitalized, they do not visit their home or go to their job for the duration of the infectious period - effectively removing them from the equation. The duration in the infectious period is equal to Normal(16, 4) time steps for mild cases, and Normal(36,8) time steps for severe and critical cases - about 8 days and 18 days respectively. From here, agents move into the recovered class, where they are unable to contract the disease. They remain here for a user-specified amount of time, at which point they become "susceptible" again. These numbers differ slightly in reality, but are generally in line with what is observed (excluding the recovered period, which is much shorter here for the sake of observing equilibria). 

The planner can execute "mask" and "vaccinate" at any point in the simulation. When "mask" is executed, all agents put on masks at some constant probability (can vary in the instance files). When "vaccinate" is executed, all agents vaccinate based on real observed probabilities relative to the agent's age bracket. An agent wearing a mask reduces its chance of contracting and transmitting, whereas vaccinating only reduces its chance of contracting. The goal function that is attempting to be optimized for this RDDL simulation is as follows:

goal = (# of masked agents * mask_penalty) 
		   + (# of vaccinated agents * vaccine_penalty) 
          + (# of infected agents * infected_penalty)
          + (# of regular beds occupied over capacity * regular_bed_penalty)
          + (# of ICU beds occupied over capacity * ICU_bed_penalty)

All of these penalties are negative, and so the goal can be a maximum of 0. Thus, by the planner maximizing this equation, the optimal series of actions should be found.

## Instructions

Clone the repo by executing the following command, or manually:

```
git clone https://github.com/fhaghighi/DiRA.git
```

From here, create and enter a [virtual environment](https://docs.python.org/3/library/venv.html), and install requirements as follows:

```
pip3 install -r requirements.txt
```

### Running Simulations

Simulations are run by executing the Sim.py file. In addition to running the simulations, this file will call upon Kingston_Info.py, which is the file that generates problem files. Please ensure that all arguments outlined in these files are as desired - the default values are the ones used that generated the data in the paper. The number of iterations and trials are both set to 20 by default. Lowering these values speeds up execution time greatly and should be done for testing purposes.

In the Sim.py, the user must specify folders that files will be read from and saved to. This must be edited directly in the Sim.py file itself, and these folders can be named whatever. Please ensure that every entry of "folder_list" directly corresponds to every entry of "domain_list", and that they are the same length.

This file must initially be run in the "Init" mode (which is the default value for the --mode argument). In addition to running simulations as specified, this will generate and save problem files for future use. This should be run every time you wish to generate new problem files. Typically, this is done when creating a "base model" that will be compared to by other models. "folder_list" represents the target folders that simulation data will be saved to, in addition to the problem files. "domain_list" is a list of domains that will be used for each folder's simulations.

From there Sim.py can be run in "Test" mode. This mode works by drawing upon previously generated problem files and seeds, and using those for the simulations. "folder_list" still represents the target folders, and should be edited as such. Likewise, "domain_list" works the same as before. This time "source_file_folder" must be changed to whatever the folder is that contains the problem files and seeds. This can only be one folder - hence, all folders in "folder_list" will be drawing from this one folder's problem files.

Sim.py has some default folders and domains written as an example, and this can be run by executing the following:

```
python3 Sim.py --mode Init --iters 2 --trials 2
```

As a result of this, a folder named "test_base_folder" will have been created that has two subfolders: Problems, iter_0, and iter_1. Problems contains the generated problem files (only 2 here, since the --iters value was set to 2), and the iter folders contain images showing what happened in each trial visually (examples located in the Images folder of this repo) in addition to a text file providing a brief summary of all trials, and a json file containing all data produced by each trial.

To run the file in "Test" mode, the lists that are commented out below "folder_list" and "domain_list" can be uncommented, and the old lists can be removed. Then, the following can be run:

```
python3 Sim.py --mode Test --iters 2 --trials 2
```

This will generate two more folders containing iteration data as previously described. This general process can be followed using any combination of arguments, within reason.

### Analysis

The other files in this repo have to do with analyzing the data produced by simulations.

Analysis.py compares the results obtained via multiple simulations with each other, and most importantly, with the model that the simulations were based upon. Note that this file ideally should be run after having generated data for models with intervention and self-isolation, intervention but no self-isolation, no intervention but self-isolation, and neither intervention nor self-isolation. There are sample folders in this repo with the necessary data to run this immediately (all folders beginning with "rec2"). The number of iterations run for each model must be changed in the Analysis.py to match what is in the source folders. "Results_rec2.txt", "Results_rec14.txt", and "Results_rec28.txt" were generated from this.

Benchmark_Graphy.py analyzes data generated by running a model with a recovery rate of 14 with no intervention, self-isolation, or hospitalization for 1000 trials on one problem file. The json file produced from this is read in, and the maximum and minimum values for agents in each SEIR class over the 1000 trials is calculated at each time step, and graphed alongside class data from a sample of 100 individual trials. Also on each graph is what is expected theoretically through using an ODE approach to modeling. The data fueling this is in ODE_Sim.csv, which is produced by ODE_Sim.Rmd. The images produced are located in the Images folder in this repo, and are titled "Max_Mins_Susceptible_SEIR.png", "Max_Mins_Exposed_SEIR.png", "Max_Mins_Infectious_SEIR.png", "Max_Mins_Recovered_SEIR.png", and "Max_Mins_SEIR.png".

