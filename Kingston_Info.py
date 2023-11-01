#Call this file when generating problem files

import osmnx as ox
import random

#LOCATION_TYPES = ["residential", "work", "commercial", "education"]

PLACES_OF_INTEREST = {
    'residential': {'residential', 'house', 'apartments', 'dormitory'},
    'work': {'office', 'commercial', 'industrial', 'retail', 'warehouse'},
    'commercial': {'commercial', 'retail'},
    'education': {'school', 'college', 'university'},
}

city = 'Kingston, Ontario, Canada'

#Sets the centroids for when determining residence/school affiliations
queens_main_centroid = (-76.495362, 44.225724)
queens_west_centroid = (-76.515323, 44.226913)
slc_centroid = (-76.527910, 44.223611)
rmc_centroid = (-76.468120, 44.232918)

#Set the university residence populations
queens_residence_pop = 25
slc_residence_pop = 3
rmc_residence_pop = 10

#Set overall Kingston population for apartment building capacity calculation (to be changed)
kingston_pop = 100

#Get the buildings
def fetch_buildings(loc):
    """Fetch the buildings from the given location."""
    buildings = ox.features_from_place(loc, tags={"building": True})
    return buildings

#Get places of interest
def fetch_places_of_interest():

    """Fetch the places of interest from the given location."""
    buildings = fetch_buildings(city)
    places_of_interest = {}
    for key, values in PLACES_OF_INTEREST.items():
        places_of_interest[key] = []
        # iterate over the buildings that have a building type in the values
        for index, building in buildings[buildings["building"].isin(values)].iterrows():
            # Grab the centroid for the place of interest
            centroid = building.geometry.centroid
            # Add the name and centroid to the list
            places_of_interest[key].append([building["building"], (centroid.x, centroid.y)])
    return places_of_interest

#Organize all locations so that they are usable for the RDDL instance file creation
#Takes the list of locations output by fetch_places_of_interest()
#Returns a sample of all locations in a dictionary that have the following structure:
#   residential: a list with each element as such -> [type of dwelling, (longitude, latitude), building capacity (only applicable for dorms and apartments), affiliation (queens, slc, rmc, or general), unique tag, distance to closest post-secondary school]
#   education: a list with each element as such -> [type of school, (longitude, latitude), affiliation (queens, slc, rmc, or regular), unique tag]
#   work: a list with each element as such -> [type of workplace, (longitude, latitude), unique tag]
#   commercial: a list with each element as such -> [type of commercial building, (longitude, latitude), unique tag]
def organize_locs(all_locs):

    #Initializing the housing lists
    queens_res_list = []
    slc_res_list = []
    rmc_res_list = []
    apartment_list = []
    other_home_list = []

    #Will count the number of residential buildings for unique tag assignment (currently unused, but might be useful later)
    home_counter = 0

    for item in all_locs['residential']:

        #Calculate the euclidian distance from a dorm to the centre of each campus
        d_to_queens_main = ox.distance.euclidean(item[1][0], item[1][1], queens_main_centroid[0], queens_main_centroid[1])
        d_to_queens_west = ox.distance.euclidean(item[1][0], item[1][1], queens_west_centroid[0], queens_west_centroid[1])
        d_to_slc = ox.distance.euclidean(item[1][0], item[1][1], slc_centroid[0], slc_centroid[1])
        d_to_rmc = ox.distance.euclidean(item[1][0], item[1][1], rmc_centroid[0], rmc_centroid[1])

        #Home is closest to Queen's
        if (d_to_queens_main==min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc) or d_to_queens_west== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):

            #If it is a dorm, then assign the Queen's residence population to it
            if (item[0] == 'dormitory'):
                item.append(queens_residence_pop)
                queens_res_list.append(item)
            
            #If it is a dorm, then assign the Kingston residence population to it
            elif (item[0] == 'apartments'):
                item.append(kingston_pop)
                apartment_list.append(item)

            #Otherwise, just assign 1 (building is a regular house)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("queens")
            item.append(home_counter)
            item.append(d_to_queens_main)

        #Home is closest to SLC
        elif (d_to_slc== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):


            #If it is a dorm, then assign the SLC's residence population to it
            if (item[0] == 'dormitory'):
                item.append(slc_residence_pop)
                slc_res_list.append(item)
            elif (item[0] == 'apartments'):
                item.append(kingston_pop)
                apartment_list.append(item)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("slc")
            item.append(home_counter)
            item.append(d_to_slc)

        #Home is closest to RMC
        elif (d_to_rmc== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):

            #If it is a dorm, then assign the RMC's residence population to it
            if (item[0] == 'dormitory'):
                item.append(rmc_residence_pop)
                rmc_res_list.append(item)
            elif (item[0] == 'apartments'):
                item.append(kingston_pop)
                apartment_list.append(item)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("rmc")
            item.append(home_counter)
            item.append(d_to_rmc)
    
        home_counter += 1

    #Takes samples of the residential buildings
    #It is done this way to ensure that residence buildings for each campus is present in a meaningful way.
    #Simply taking a sample of the entire residential building list could leave out important residences
    queens_res_list = random.sample(queens_res_list, 4)
    rmc_res_list = random.sample(rmc_res_list, 1)
    slc_res_list = slc_res_list
    apartment_list = random.sample(apartment_list, 5)
    other_home_list = random.sample(other_home_list, 10)

    #Assigning buildings capacity according to the user-specified populations
    for item in all_locs['residential']:
        if (item[0] == 'dormitory'):
            if (item[3] == 'queens'):
                item[2] = item[2]//len(queens_res_list)
            elif (item[3] == 'slc'):
                item[2] = item[2]//len(slc_res_list)
            elif (item[3] == 'rmc'):
                item[2] = item[2]//len(rmc_res_list)
        elif (item[0] == 'apartments'):
            item[2] = (item[2] - queens_residence_pop - slc_residence_pop - rmc_residence_pop - (2.5 * len(other_home_list)))//len(apartment_list)

    #Performing a very similar task here, but with education buildings
    queens_edu_list = []
    slc_edu_list = []
    rmc_edu_list = []
    other_edu_list = []

    #Job counter is used for both education environments and workplaces. This is because an agent will either go to school or work, and so we give them unique tags simultaneously
    job_counter = 0
    for item in all_locs['education']:
        if (item[0] == 'college'):
            item.append("slc")
            item.append(job_counter)
            slc_edu_list.append(item)
        
        elif (item[0] == 'university'):
            d_to_queens_main = ox.distance.euclidean(item[1][0], item[1][1], queens_main_centroid[0], queens_main_centroid[1])
            d_to_queens_west = ox.distance.euclidean(item[1][0], item[1][1], queens_west_centroid[0], queens_west_centroid[1])
            d_to_rmc = ox.distance.euclidean(item[1][0], item[1][1], rmc_centroid[0], rmc_centroid[1])

            if (d_to_rmc== min(d_to_rmc, d_to_queens_main, d_to_queens_west)):
                item.append("rmc")
                item.append(job_counter)
                rmc_edu_list.append(item)
            
            else:
                item.append("queens")
                item.append(job_counter)
                queens_edu_list.append(item)
        
        else:
            item.append("regular")
            item.append(job_counter)
            other_edu_list.append(item)

        job_counter += 1

    store_counter = 0
    for item in all_locs['commercial']:

        item.append(store_counter)
        store_counter += 1

    for item in all_locs['work']:

        item.append(job_counter)
        job_counter += 1

    #Once again, samples are taken in this way to ensure that post-secondary education buildings are present
    queens_edu_list = random.sample(queens_edu_list, 5)
    slc_edu_list = random.sample(slc_edu_list, 1)
    rmc_edu_list = random.sample(rmc_edu_list, 2)   
    other_edu_list = random.sample(other_edu_list, len(other_edu_list) // 3)

    commercial_list = random.sample(all_locs['commercial'], len(all_locs['commercial']) // 5)
    work_list = random.sample(all_locs['work'], 4)

    #Assigning values to the keys according to the random samples generated above
    all_locs['residential'] = queens_res_list + rmc_res_list + slc_res_list + apartment_list + other_home_list
    all_locs['education'] = queens_edu_list + rmc_edu_list + slc_edu_list + other_edu_list
    all_locs['work'] = work_list
    all_locs['commercial'] = commercial_list

    return all_locs

#Takes in all residential buildings, all queens/slc/rmc educational buildings, all workplaces, and the student + general populations 
#Returns a list of items with the following format: [[agent, age bracket, dorm, school building], dorm coordinates, school building coordinates]
#TO DO: Add school assignments for ages 0-19
def assign_agents(all_housing, queens_buildings, slc_buildings, rmc_buildings, workplaces, student_pops, general_pop):

    #Will be returned at the end
    agent_homes = []

    #Keeps track of the agents and homes for naming purposes
    agent_counter = 0
    home_counter = 0

    #Each list holds residential buildings that are closest to a given campus
    queens_homes = []
    slc_homes = []
    rmc_homes = []

    for home in all_housing:
        if home[3] == "queens":
            queens_homes.append(home)
        elif home[3] == "slc":
            slc_homes.append(home)
        elif home[3] == "rmc":
            rmc_homes.append(home)

    #Loops through the populations for each school (position 0: Queen's, position 1: SLC, position 2: RMC)
    for i in range (0, len(student_pops)):

        #While there are still more students to be assigned
        while student_pops[i] > 0:

            #Selects the residential building closest to a given campus. Generally selects dorms first
            if (i == 0):
                housing = min(queens_homes, key=lambda x: x[5])
            elif (i == 1):
                housing = min(slc_homes, key=lambda x: x[5])
            else:
                housing = min(rmc_homes, key=lambda x: x[5])

            #The capacity for a given residential building
            cap = housing[2]

            #Loops while a building is still under capacity and students still need to be assigned
            while (cap > 0 and student_pops[i] > 0):

                #Each home can hold 1-4 students
                current_in_house = random.randint(1, 4)

                for j in range (current_in_house):

                    #Assigns the agent a building based on what campus the residential building is nearest to
                    if housing[3] == "queens":
                        job = random.choice(queens_buildings)
                        student_pops[0] -= 1
                    
                    elif housing[3] == "slc":
                        job = random.choice(slc_buildings)
                        student_pops[1] -= 1

                    elif housing[3] == "rmc":
                        job = random.choice(rmc_buildings)
                        student_pops[2] -= 1
                    
                    #Appends everything to the student_homes list
                    agent_homes.append([["a" + str(agent_counter), 2, home_counter, job[3]], housing[1], job[1]])

                    #Increases the agent #, and decreases the capacity
                    cap -= 1
                    agent_counter += 1

                #Changes the unique home value
                home_counter += 1

            #For each campus option, if the capacity is less than 0, remove that residential building from the list. Otherwise, change its capacity to whatever the current capacity is
            #This is so that apartments can be partially filled by students, with the remaining units possibly being filled up by other people in the next section
            if (i == 0):
                if (cap <= 0):
                    queens_homes.remove(min(queens_homes, key=lambda x: x[5]))
                else:
                    min(queens_homes, key=lambda x: x[5])[2] = cap
            elif (i == 1):
                if (cap <= 0):
                    slc_homes.remove(min(slc_homes, key=lambda x: x[5]))
                else:
                    min(slc_homes, key=lambda x: x[5])[2] = cap
            else:
                if (cap <= 0):
                    rmc_homes.remove(min(rmc_homes, key=lambda x: x[5]))
                else:
                    min(rmc_homes, key=lambda x: x[5])[2] = cap
            
    #Combine these residential buildings back into one list, after deleting items and changing capacity values
    all_homes = queens_homes + slc_homes + rmc_homes

    #Remove dorms for the next section
    for item in all_homes:
        if item[0] == 'dormitory':
            all_homes.remove(item)

    random.shuffle(all_homes)

    #Loops while there are still people left to assign
    while general_pop > 0:

        #Choose a random home from the list of all homes
        home = random.choice(all_homes)

        cap = home[2]

        #Loops while a building is still under capacity and people still need to be assigned
        while (cap > 0 and general_pop > 0):

            #Each home can hold 1-4 people (to be changed to incorporate probabilities)
            current_in_house = random.randint(1, 4)

            for j in range (current_in_house):

                #Assigns the agent a workplace
                job = random.choice(workplaces)
                general_pop -= 1
                
                #Appends everything to the student_homes list
                agent_homes.append([["a" + str(agent_counter), 4, home_counter, job[2]], home[1], job[1]])

                #Increases the agent #, and agents currently occupying a building
                cap -= 1
                agent_counter += 1

            #Changes the unique home value
            home_counter += 1

        #Home always removed. Even if a building is not totally occupied, no more agents will be assigned anyways, so it can just be removed
        all_homes.remove(home)

    return agent_homes

#Assigns agents stores to visit on the weekend
#Returns a list of items that are of the following format: [[agent, age bracket, dorm, school building, store], dorm coordinates, school building coordinates, store coordinates]
def assign_stores(agent_homes_work, stores_list):

    for agent in agent_homes_work:

        #For each agent, loops through all stores to find which one is the closest
        min_d_to_store = ox.distance.euclidean(agent[1][0], agent[1][1], stores_list[0][1][0], stores_list[0][1][1])
        coords = stores_list[0][1]
        assignment = stores_list[0][2]

        for store in stores_list:
            d_to_store = ox.distance.euclidean(agent[1][0], agent[1][1], store[1][0], store[1][1])
            if d_to_store <= min_d_to_store:
                min_d_to_store = d_to_store
                coords = store[1]
                assignment = store[2]
        
        #Assigns the closest store to the agent
        agent[0].append(assignment)
        agent.append(coords)

    return agent_homes_work


locs = fetch_places_of_interest()

organized_locs = organize_locs(locs)

#Retrieves all dorms
all_residences = []
other_homes = []

for item in organized_locs['residential']:
    if (item[0] == 'dormitory'):
        all_residences.append(item)
    
    else:
        other_homes.append(item)

#Retrieves all education buildings of different types
all_schools = []
queens = []
slc = []
rmc = []

for item in organized_locs['education']:
    if (item[0] == 'school'):
        all_schools.append(item)
    elif (item[2] == 'queens'):
        queens.append(item)
    elif (item[2] == 'slc'):
        slc.append(item)
    elif (item[2] == 'rmc'):
        rmc.append(item)

#Assign school + general populations
school_populations = [30, 5, 15]
general_population = 10

#Calls the functions
student_homes_school = assign_agents(organized_locs['residential'], queens, slc, rmc, organized_locs['work'], school_populations, general_population)

agent_complete = assign_stores(student_homes_school, organized_locs['commercial'])


#Writes info to RDDL instance file

with open('instance.rddl', 'w') as f:
    f.write ('non-fluents covid-sim_nf_1 {')
    f.write('\n')
    f.write('\tdomain = covid-sim;')
    f.write('\n')
    f.write('\tobjects {')
    f.write('\n')

    agent_string = 'agent : {'
    for agent in agent_complete:
        agent_string = agent_string + agent[0][0] + ', '
    agent_string = agent_string[:-2] + '};'

    f.write('\t\t' + agent_string)

    f.write('\n')
    f.write('\t};')
    f.write('\n')
    f.write('\tnon-fluents {')
    f.write('\n')

    f.write

    for agent in agent_complete:
        f.write('\t\tAGENT_AGE(' + agent[0][0] + ') = ' + str(agent[0][1]) + ';')
        f.write('\n')
        f.write('\t\tAGENT_HOME(' + agent[0][0] + ') = ' + str(agent[0][2]) + ';')
        f.write('\n')
        f.write('\t\tAGENT_JOB(' + agent[0][0] + ') = ' + str(agent[0][3]) + ';')
        f.write('\n')
        f.write('\t\tAGENT_STORE(' + agent[0][0] + ') = ' + str(agent[0][4]) + ';')
        f.write('\n')
    
    f.write('\t};')
    f.write('\n')
    f.write('}')
    f.write('\n')
    f.write('\n')

    f.write('instance covid-sim_inst_1 {')
    f.write('\n')
    f.write('\tdomain = covid-sim;')
    f.write('\n')
    f.write('\tnon-fluents = covid-sim_nf_1;')
    f.write('\n')
    f.write('\tinit-state {')
    f.write('\n')

    for agent in agent_complete:
        infectious = random.randint(1,10)

        if infectious == 1:
            f.write('\t\tinfectious(' + agent[0][0] + ');')
            f.write('\n')
        
    f.write('\t};')
    f.write('\n')
    f.write('\n')
    f.write('\thorizon = 50;')
    f.write('\n')
    f.write('\tdiscount = 1.0;')
    f.write('\n')
    f.write('}')



