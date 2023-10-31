# Modeling Disease Spread in a Community

This RDDL domain models the spread of disease in a community. Agents, in this domain, move through the following SEIR states:

 - Susceptible: An agent does not have the disease, and has no natural immunity to it
 - Exposed: An agent has contracted the disease, but is asymptomatic and unable to spread the disease to other agents
 - Infectious: An agent is showing symptoms for the disease, and is able to spread it to other agents
 - Recovered: An agent is recovered from the disease, and is unable to contract it

Before describing disease dynamics in detail, I will explain how the agents are initialized and how they move around. Agents have the following properties associated with them:

 - Age Bracket (Ages are evenly distributed for the sake of simplicity)
     - 0 = ages 0-9, 1 = ages 1-19, and so on
 - Home (where an agent goes every night)
	 - Only 1-4 agents per home
 - Job (where an agent goes during the day on weekdays. Children can be thought of as going to school and adults as going to work)
 - Store (where an agent goes during the day on weekends)

Each day is broken up into two time steps - day and night. On the "day" time step, all agents go to their jobs, and on the "night" time step, all agents go to their homes (these can change when isolating or hospitalized - more on this later). This is tracked by a simple counter, ticking between 0 and 1. Similarly, a counter is used to track weekdays and weekends. Weekdays are represented by counter values 1-10, and weekends are represented by counter values 11-14. Note that traveling between points is not captured here - the agents simply teleport around.

Agents, by default, begin in the susceptible state (in the domain, susceptible agents are simply those who are neither exposed, infectious, or recovered). When an agent is in the susceptible state, they move around the world freely. When they are at a location, they contract the disease at a probability of (# of infectious agents at location / total # of agents at a location). If they do contract the disease, then they are moved into the exposed class, where they remain for a total of Normal(9,2) time steps (about 4.5 days). From here, they move into the infectious class. When they move into the infectious class, they have a probability of isolating at home (equal to the probability of them wearing a mask). If they choose to isolate, they do not go to their job, and simply remain at home (they can still infect other members of their household). Their infection can also be mild, severe, or critical. In the mild case, it is business as usual. In the severe and critical cases, they become hospitalized. If the infection is severe, then they occupy a regular hospital bed. If the infection is critical, then they occupy an ICU bed. Both of these have limits, and exceeding these limits results in a penalty (more on this later). If an agent is hospitalized, they do not visit their home or go to their job for the duration of the infectious period - effectively removing them from the equation. The duration in the infectious period is equal to Normal(16, 4) time steps for mild cases, and Normal(36,8) time steps for severe and critical cases - about 8 days and 18 days respectively. From here, agents move into the recovered class, where they are unable to contract the disease. They remain here for Normal(15,3) time steps, or about 7.5 days, at which point they become "susceptible" again. These numbers differ slightly in reality, but are generally in line with what is observed (excluding the recovered period, which is much shorter here for the sake of observing equilibria). 

The planner can execute "mask" and "vaccinate" at any point in the simulation. When "mask" is executed, all agents put on masks at some constant probability (can vary in the instance files). When "vaccinate" is executed, all agents vaccinate based on real observed probabilities relative to the agent's age bracket. "vaccine" can only be executed one time, at which point, the agents who take the vaccine are vaccinated for the whole simulation. An agent wearing a mask reduces its chance of contracting and transmitting, whereas vaccinating only reduces its chance of contracting. The goal function that is attempting to be optimized for this RDDL simulation is as follows:

goal = (# of masked agents * mask_penalty) 
		   + (# of vaccinated agents * vaccine_penalty) 
          + (# of infected agents * infected_penalty)
          + (# of regular beds occupied over capacity * regular_bed_penalty)
          + (# of ICU beds occupied over capacity * ICU_bed_penalty)

In the future, the number of infected agents will have no bearing on the goal. The probabilities that an agent is hospitalized are so low that it is unlikely for any agents to be hospitalized given the number of agents used per simulation. So, I chose to have the number of infected agents to count towards the goal so that actions would be taken by the planner.

All of these penalties are negative, and so the goal can be a maximum of 0. Thus, by the planner maximizing this equation, the optimal series of actions should be found.

I have included four instance files that are similar, but slightly different with regards to the transmission chances and action penalties. When running the simulation, the important parts of the visualization to look at are "susceptible_count", "exposed_count", "infectious_count", "recovered_count", "vaccine_implemented", and "mask_implemented". There are too many agents for many of the other charts to be readable.

In instance_1.rddl, the mask and vaccine penalties are the same. However, due to the chance of an agent wearing a mask being 0.8 (compared to the vaccine chance being over 0.8 on average for the age brackets), and the fact that masks reduce transmission less than vaccines, the planner only implements the vaccine mandate, and never introduces mask mandates.

In instance_3.rddl, the only change is that the mask penalty changes from -10 to -4. With this, the planner implements the mask mandate throughout the simulation (note that if the mask penalty is -5, it does not implement it at all).

In instance_2.rddl, the masks and vaccines were made much more effective, and the chance of masking is now 1. The mask and vaccine penalties are -20. With this setup, the planner takes both the "mask" and "vaccinate" actions immediately, and they are active throughout the simulation. 

In instance_4.rddl, the mask and vaccine penalties were changed to be -200 each. In this case, neither "mask" nor "vaccine" is ever taken, and the simulation runs freely.