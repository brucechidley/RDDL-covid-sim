#Author: Bruce Chidley
#Compiles reward data and compares results between models

import numpy as np
import json
import matplotlib.pyplot as plt

#CHANGE FOLDERS AS NEEDED
#These folders will be where the data is drawn from
base_folder = ["rec2_no_intervention_no_iso"]
isolation_folders = ["rec2_no_intervention_iso_03", "rec2_no_intervention_iso_05", "rec2_no_intervention_iso_07"]
intervention_folders = ["rec2_iso_03", "rec2_iso_05", "rec2_iso_07"]
no_iso_folder = ["rec2_no_iso"]

#The name of the base model (rec2, rec14, or rec28) (CHANGE AS NEEDED)
model_name = "rec2"

all_rewards_base = []
average_rewards_base = []

all_rewards_isolation = []
average_rewards_isolation = []

all_rewards_no_iso = []
average_rewards_no_iso = []

all_rewards_intervention = []
average_rewards_intervention = []

#CHANGE AS NEEDED
#This is the number of iterations that each model was run on
iter = 2


#Appends all rewards for each iterations (20 rewards for the 20 total trials) to a 2D list
#Also appends average rewards for each iteration to a 2D list

for i in range (0, iter):

    f = open(str(base_folder[0]) + "/iter_" + str(i) + "/data_iter_" + str(i) + ".json")
    data = json.load(f)

    rewards = np.array(eval(data["All_rewards"]))

    all_rewards_base.append(rewards)

    average_rewards_base.append(np.average(rewards))

for folder in isolation_folders:
    rewards_average_current = []
    all_rewards_current = []
    for i in range (0, iter):
        f = open(str(folder) + "/iter_" + str(i) + "/data_iter_" + str(i) + ".json")
        data = json.load(f)

        rewards = np.array(eval(data["All_rewards"]))

        all_rewards_current.append(rewards)

        rewards_average_current.append(np.average(rewards))

    all_rewards_isolation.append(all_rewards_current)
    average_rewards_isolation.append(rewards_average_current)

for i in range (0, iter):

    f = open(str(no_iso_folder[0]) + "/iter_" + str(i) + "/data_iter_" + str(i) + ".json")
    data = json.load(f)

    rewards = np.array(eval(data["All_rewards"]))

    all_rewards_no_iso.append(rewards)

    average_rewards_no_iso.append(np.average(rewards))

for folder in intervention_folders:
    rewards_average_current = []
    all_rewards_current = []
    for i in range (0, iter):
        f = open(str(folder) + "/iter_" + str(i) + "/data_iter_" + str(i) + ".json")
        data = json.load(f)

        rewards = np.array(eval(data["All_rewards"]))

        all_rewards_current.append(rewards)

        rewards_average_current.append(np.average(rewards))

    all_rewards_intervention.append(all_rewards_current)
    average_rewards_intervention.append(rewards_average_current)

#Compiles all data into one list
data = [average_rewards_base, 
        average_rewards_isolation[0], average_rewards_isolation[1], average_rewards_isolation[2], 
        average_rewards_no_iso,
        average_rewards_intervention[0], average_rewards_intervention[1], average_rewards_intervention[2]]

#These will be the names of the models on the x axis
names = ["Base", 
         "OSI 0.3", "OSI 0.5", "OSI 0.7",
         "OIn",
         "InSI 0.3", "InSI 0.5", "InSI 0.7"]

#Create box plots for each model
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(names, fontsize=10)
ax.ticklabel_format(axis='y', style='plain')

save_name = model_name + "_BoxPlot.png"
plt.title("Rewards for a Recovery Rate of 2 Time Steps")
plt.xlabel("Model Configuration", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.savefig(save_name, bbox_inches="tight")
plt.clf()


#Gets the average reward across all iterations

total_average_reward_base = np.average(average_rewards_base)
total_average_rewards_no_iso = np.average(average_rewards_no_iso)

all_average_rewards_isolation = []
all_average_rewards_intervention = []

for i in range (0, len(intervention_folders)):

    all_average_rewards_isolation.append(np.average(average_rewards_isolation[i]))
    all_average_rewards_intervention.append(np.average(average_rewards_intervention[i]))


#Divides to get reward ratios

iso_to_base_div_total = []
intervention_to_base_div_total = []
intervention_to_iso_div_total = []
intervention_to_no_iso_div_total = []

no_iso_to_base_div_total = np.divide(total_average_rewards_no_iso, total_average_reward_base)

for i in range (0, len(intervention_folders)):

    iso_to_base_div_total.append(np.divide(all_average_rewards_isolation[i], total_average_reward_base))
    intervention_to_base_div_total.append(np.divide(all_average_rewards_intervention[i], total_average_reward_base))
    intervention_to_iso_div_total.append(np.divide(all_average_rewards_intervention[i], all_average_rewards_isolation[i]))
    intervention_to_no_iso_div_total.append(np.divide(all_average_rewards_intervention[i], total_average_rewards_no_iso))

#Write the percent improvements to a txt file
with open("Results_" + model_name + ".txt", 'w') as f:
    f.write("Improvements over base model using self-isolation and no intervention:")
    f.write("\n")
    f.write ("Self-Isolation at rate 0.3: " + str(round((1 - iso_to_base_div_total[0]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.5: " + str(round((1 - iso_to_base_div_total[1]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.7: " + str(round((1 - iso_to_base_div_total[2]), 3)))
    f.write('\n')
    f.write('\n')

    f.write("Improvement over base model using intervention and no self-isolation: " + str(round((1 -no_iso_to_base_div_total), 3)))
    f.write("\n")
    f.write("\n")

    f.write("Improvements over base model using self-isolation and intervention:")
    f.write("\n")
    f.write ("Self-Isolation at rate 0.3: " + str(round((1 - intervention_to_base_div_total[0]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.5: " + str(round((1 - intervention_to_base_div_total[1]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.7: " + str(round((1 - intervention_to_base_div_total[2]), 3)))
    f.write('\n')
    f.write('\n')

    f.write("Improvements over model with only self-isolation using self-isolation and intervention:")
    f.write("\n")
    f.write ("Self-Isolation at rate 0.3: " + str(round((1 - intervention_to_iso_div_total[0]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.5: " + str(round((1 - intervention_to_iso_div_total[1]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.7: " + str(round((1 - intervention_to_iso_div_total[2]), 3)))
    f.write('\n')
    f.write('\n')

    f.write("Improvements over model with only intervention using self-isolation and intervention:")
    f.write("\n")
    f.write ("Self-Isolation at rate 0.3: " + str(round((1 - intervention_to_no_iso_div_total[0]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.5: " + str(round((1 - intervention_to_no_iso_div_total[1]), 3)))
    f.write('\n')
    f.write ("Self-Isolation at rate 0.7: " + str(round((1 - intervention_to_no_iso_div_total[2]), 3)))
    f.write('\n')
    f.write('\n')