#Author: Bruce Chidley
#Graphs data from the benchmark model simulations
#Plots the max and min values for each class at each time step over 1000 trials alongside the ODE model output

import numpy as np
import json
import matplotlib.pyplot as plt
import csv
import random

#Opens json file for the benchmark model (recovery rate 14 with no intervention, isolation, or hospitalization)
f = open('rec14_benchmark/iter_0/data_iter_0.json')

data = json.load(f)

#Returns the count of agents in each state at every time step for every trial
all_susceptible = np.array(eval(data['Susceptible']))
all_exposed = np.array(eval(data['Exposed']))
all_infectious = np.array(eval(data['Infectious']))
all_recovered = np.array(eval(data['Recovered']))

#Gives counts for each step of the ODE model, created in R
ODE_time = []
ODE_susceptible = []
ODE_exposed = []
ODE_infectious = []
ODE_recovered = []

with open('ODE_Sim.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        ODE_time.append(row[0])
        ODE_susceptible.append(row[1])
        ODE_exposed.append(row[2])
        ODE_infectious.append(row[3])
        ODE_recovered.append(row[4])

#Removes the column names
ODE_time.pop(0)
ODE_susceptible.pop(0)
ODE_exposed.pop(0)
ODE_infectious.pop(0)
ODE_recovered.pop(0)

#Converts to lists of floats
ODE_time = list(np.float_(ODE_time))
ODE_susceptible = list(np.float_(ODE_susceptible))
ODE_exposed = list(np.float_(ODE_exposed))
ODE_infectious = list(np.float_(ODE_infectious))
ODE_recovered = list(np.float_(ODE_recovered))

#Returns 100 values between 0 and 999
#These will be the samples used for the graph generation
sample = random.sample(range(0, 1000), 100)
time_step = np.array(eval(data['Time_step']))[0]


#Calculates the ranges for each class population at each time step
min_susceptible = []
min_exposed = []
min_infectious = []
min_recovered = []

max_susceptible = np.max(all_susceptible, 0)
max_exposed = np.max(all_exposed, 0)
max_infectious = np.max(all_infectious, 0)
max_recovered = np.max(all_recovered, 0)

min_susceptible = np.min(all_susceptible, 0)
min_exposed = np.min(all_exposed, 0)
min_infectious = np.min(all_infectious, 0)
min_recovered = np.min(all_recovered, 0)

#For each graph:
#Plot the 100 samples on the same graph
#Overlay the max and min values for each class
#Overlay the ODE model output for each class
#Save the graph as a png file

for i in sample:
    plt.plot(time_step, all_susceptible[i], color='palegreen', linewidth=0.5)

plt.plot(time_step, max_susceptible, color='green', label='Max/Min Susceptible')
plt.plot(time_step, min_susceptible, color='green')
plt.plot(time_step, ODE_susceptible, color='black', label="ODE Susceptible")
plt.title('Susceptible Agents Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of Agents")
plt.legend()

save_name = "Max_Mins_Susceptible_SEIR.png"
plt.savefig(save_name)
plt.clf()

for i in sample:
    plt.plot(time_step, all_exposed[i], color='navajowhite', linewidth=0.5)

plt.plot(time_step, max_exposed, color='orange', label='Max/Min Exposed')
plt.plot(time_step, min_exposed, color='orange')
plt.plot(time_step, ODE_exposed, color='black', label="ODE Exposed")
plt.title('Exposed Agents Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of Agents")
plt.legend()

save_name = "Max_Mins_Exposed_SEIR.png"
plt.savefig(save_name)
plt.clf()

for i in sample:
    plt.plot(time_step, all_infectious[i], color='lightcoral', linewidth=0.5)

plt.plot(time_step, max_infectious, color='red', label='Max/Min Infectious')
plt.plot(time_step, min_infectious, color='red')
plt.plot(time_step, ODE_infectious, color='black', label="ODE Infectious")
plt.title('Infectious Agents Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of Agents")
plt.legend()

save_name = "Max_Mins_Infectious_SEIR.png"
plt.savefig(save_name)
plt.clf()

for i in sample:
    plt.plot(time_step, all_recovered[i], color='cornflowerblue', linewidth=0.5)

plt.plot(time_step, max_recovered, color='blue', label='Max/Min Recovered')
plt.plot(time_step, min_recovered, color='blue')
plt.plot(time_step, ODE_recovered, color='black', label="ODE Recovered")
plt.title('Recovered Agents Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of Agents")
plt.legend()

save_name = "Max_Mins_Recovered_SEIR.png"
plt.savefig(save_name)
plt.clf()


#Plot the maxes and mins for all classes on one graph and save it
plt.plot(time_step, max_susceptible, color='green', label='Susceptible')
plt.plot(time_step, min_susceptible, color='green')
plt.plot(time_step, max_exposed, color='orange', label='Exposed')
plt.plot(time_step, min_exposed, color='orange')
plt.plot(time_step, max_infectious, color='red', label='Infectious')
plt.plot(time_step, min_infectious, color='red')
plt.plot(time_step, max_recovered, color='blue', label='Recovered')
plt.plot(time_step, min_recovered, color='blue')
plt.title('SEIR Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of Agents")
plt.legend()

save_name = "Max_Mins_SEIR.png"
plt.savefig(save_name)
plt.clf()
