import numpy as np
import json
import matplotlib.pyplot as plt

f = open('Documents/CISC_813/Project/RDDL_Sim/Images_rec14_benchmark/iter_0/data_iter_0.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

all_susceptible = np.array(eval(data['Susceptible']))
all_exposed = np.array(eval(data['Exposed']))
all_infectious = np.array(eval(data['Infectious']))
all_recovered = np.array(eval(data['Recovered']))

time_step = np.array(eval(data['Time_step']))[0]

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

save_name = "Documents/CISC_813/Project/RDDL_Sim/Max_Mins_SEIR.png"
plt.savefig(save_name)
plt.clf()