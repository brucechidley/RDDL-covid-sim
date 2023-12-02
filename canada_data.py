#Author: Bruce Chidley

import csv
import matplotlib.pyplot as plt

all_info = []

#Load in the required historical Canadian covid-19 data
with open('covid19_stats.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if not (row[0] == "pruid"):
            if (row[8] == "" or row[8] == "-"):

                #Append [province, date, week, year]
                all_info.append([row[1], row[3], int(row[4]), int(row[5]), 0])
            else:
                all_info.append([row[1], row[3], int(row[4]), int(row[5]), int(row[8])])

ontario_info = []

#Isolate ontario data
for element in all_info:
    if element[0] == "Ontario":
        ontario_info.append(element)

final_ontario_info = []
    
#Disregard data prior to August 2020
for element in ontario_info:
    if not ((element[3] == 2020 and element[2] <= 28)):
        final_ontario_info.append(element)

rolling_sums = []

#Sum weekly cases with the prior's week cases - each infectious period lasts about 2 weeks
for i in range (0, len(final_ontario_info)):
    rolling_sum = 0
    for k in range (0,2):
        index = i - k
        if (index >= 0):
            rolling_sum += final_ontario_info[index][4]
    rolling_sums.append(rolling_sum)
    
ontario_dates = []

#Get dates for the purpose of graphing
for element in final_ontario_info:
    ontario_dates.append(element[1])

#Graphs the data, showing an estimate of confirmed cases per week
plt.plot(ontario_dates, rolling_sums, color='green')
plt.title('COVID-19 Cases Over Time')
plt.xlabel("Time Steps")
plt.ylabel("Number of People")
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend()

save_name = "covid_cases_ontario.png"
plt.savefig(save_name)
plt.clf()