# import needed packages and set options
#---------------------------------------

import math
import random
import pandas as pd
import numpy  as np
random.seed(1492)

pd.set_option('display.max_rows', 100)
pd.set_option('display.width',    100)

# bring in the distances data and create a team list and dictionary distance lookup
#----------------------------------------------------------------------------------

DISTANCES = pd.read_csv('../data/StadiumDistances.csv')
TEAMLIST  = list(set([row for row in DISTANCES.OrigCode]))
DISTDICT  = {(row.OrigCode, row.DestCode): row.Distance for row in DISTANCES.itertuples()}

# create functions to build an initial random population 
#-------------------------------------------------------

def initPop(teamlist, n):
    '''create a random initial population of stadium paths to kick off the optimization algorithm
    @params:
        - teamlist[list]: list containing three-letter team codes for every MLB team
        - n[int]: number of individuals to generate for the initial population
    @return:
        - population[list]: list of randomly generated stadium paths represented themselves as lists'''
    
    population = []
    while len(population) < n:
        indiv    = []
        tosample = teamlist[:]
        while len(indiv) < len(teamlist):
            indiv.append(tosample.pop(random.randint(0, len(teamlist) - len(indiv) - 1)))
        population.append(indiv)
    return population
    
# create functions to calculate the total distance of an individual and the fitness of a population
#--------------------------------------------------------------------------------------------------
    
def calcDistance(distdict, indiv):
    '''calculate the total round-trip distance traveled on any given stadium path
    @params:
        - distdict[dict]: distance lookup dictionary where the keys are pairs of team codes
        - indiv[list]: individual stadium path for which to calculate total distance
    @return:
        - distance[int]: total round-trip distance traveled on stadium path (in meters)'''
    return sum([distdict[(indiv[i % len(indiv)], indiv[(i+1) % len(indiv)])] for i in range(len(indiv))])
    
def calcFitness(distdict, population):
    '''calculate the 'fitness' for each member of the current population of stadium paths
    an individual/path's fitness is defined as the difference between its total distance 
    and the maximum distance for any path in the population (higher is better)
    @params:
        - distdict[dict]: distance lookup dictionary where the keys are pairs of team codes
        - population[list]: list containing current population of stadium paths (as lists)
    @return:
        - fitness[list]: list of fitness values for each stadium path in the current population'''

    distances   = [calcDistance(distdict, indiv) for indiv in population]
    maxdistance = max(distances)
    return [maxdistance - dist for dist in distances]
    
def calcDistStats(distdict, population):
    '''calculate distance statistics for the current population of stadium paths
    @params:
        - distdict[dict]: distance lookup dictionary where the keys are pairs of team codes
        - population[list]: list containing current population of stadium paths (as lists)
    @return:
        - diststats[dict]: min/max/avg distance, and position of best current path'''

    distances = [calcDistance(distdict, indiv) for indiv in population]
    avgdist   = np.mean(distances)
    mindist   = np.min(distances)
    maxdist   = np.max(distances)
    bestpos   = distances.index(min(distances))
    return {'avgdist': avgdist, 'mindist': mindist, 'maxdist': maxdist, 'bestpos': bestpos}
    
# create functions to perform selection for the next generation
#--------------------------------------------------------------
    
def getParents(n, population, fitness):
    '''probabilistically sample the current population with replacement to select a set of parents 
    which will then be used to create the next generation. sampling done with stochastic acceptance: 
    p(accept) = fitness(i) / fitness(max) for randomly drawn individuals until capacity is met
    @params:
        - n[int]: number of parents to select from the current population (including duplicates)
        - population[list]: list containing current population of stadium paths (as lists)
        - fitness[list]: list of fitness values for each stadium path in the current population
    @return:
        - parents[list]: list of parents selected from the current population for the next generation'''
    
    maxindex = fitness.index(max(fitness))
    parents  = [population[maxindex]]
    
    while len(parents) < n:
        index = random.randint(0, len(population)-1)
        if random.random() < fitness[index] / fitness[maxindex]:
            parents.append(population[index])
    return parents
    
def getCouples(n, parents):
    '''generate random pairs of non-identical parents (couples) from a given parent set
    @params:
        - n[int]: number of couples (sets of two different parents) to create
        - parents[list]: list of parents selected from the current population for the next generation
    @return:
        - couples[list]: list of couples where each couple is stored as a 2-tuple'''

    couples = []
    while len(couples) < n:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        if p1 != p2:
            couples.append((p1, p2))
    return couples
    
# create functions to perform crossover for the next generation
#--------------------------------------------------------------
    
def crossOX(n, parents):
    '''implement the order-crossover (OX) method of producing new children from a set of parents
    a child maintains a random subtour from p1 all other elements follow the ordering of p2
    @params:
        - n[int]: number of children to create
        - parents[list]: list of parents to use to create children
    @return:
        - children[list]: new list of child paths created by applying the algorithm to the parents'''
    
    couples  = getCouples(n/2, parents)
    children = []
    
    for p1, p2 in couples:
        
        # get the start/end position of the subtour to be preserved for this couple
        
        subbeg = random.randint(0,      len(p1)-1)
        subend = random.randint(subbeg, len(p1)-1)
        
        # initialize each child with the subtour of the corresponding parent
        
        c1 = [p1[i] if (i >= subbeg and i <= subend) else None for i in range(len(p1))]
        c2 = [p2[i] if (i >= subbeg and i <= subend) else None for i in range(len(p2))]
        
        # fill the remaining slots in child 1 based on the sequence order of parent 2        
        
        c1pos1 = (subend + 1) % len(c1)
        c2pos1 = (subend + 1) % len(c2)
        
        while c1.count(None) > 0:
            if p2[c2pos1] not in c1:
                c1[c1pos1] = p2[c2pos1]
                c1pos1 = (c1pos1 + 1) % len(c1)
                c2pos1 = (c2pos1 + 1) % len(c2)
            else:
                c2pos1 = (c2pos1 + 1) % len(c2)
                
        # fill the remaining slots in child 2 based on the sequence order of parent 1   
                
        c1pos2 = (subend + 1) % len(c1)
        c2pos2 = (subend + 1) % len(c2)
        
        while c2.count(None) > 0:
            if p1[c1pos2] not in c2:
                c2[c2pos2] = p1[c1pos2]
                c1pos2 = (c1pos2 + 1) % len(c1)
                c2pos2 = (c2pos2 + 1) % len(c2)
            else:
                c1pos2 = (c1pos2 + 1) % len(c1)
                
        children.extend([c1, c2])
        
    return children
    
def crossER(n, parents):
    '''implement the edge-recombination-crossover (ER) method of producing new children
    children are created using the common edges of their parents
    @params:
        - n[int]: number of children to create
        - parents[list]: list of parents to use to create children
    @return:
        - children[list]: new list of child paths created by applying the algorithm to the parents'''

    couples  = getCouples(n, parents)
    children = []
    
    for p1, p2 in couples:
        
        child = []
        
        # create an edgelist containing connected nodes from both parents for each node
        
        edgelist = {p1[i]: set([p1[(i-1) % len(p1)], p1[(i+1) % len(p1)]]) for i in range(len(p1))}
        for key in edgelist.keys():
            edgelist[key] = edgelist[key].union(set([p2[(p2.index(key)-1) % len(p2)], p2[(p2.index(key)+1) % len(p2)]]))
            
        # initialize the child with the first node from the parent with the fewest edges (or random if tied)
            
        if   len(edgelist[p1[0]]) > len(edgelist[p2[0]]):
            child.append(p2[0])
        elif len(edgelist[p1[0]]) < len(edgelist[p2[0]]):
            child.append(p1[0])
        else:
            child.append(random.choice([p1[0], p2[0]]))
            
        # remove the chosen first node from the edgelists of every node (so it won't be considered again) 
            
        for key in edgelist.keys():
            edgelist[key].discard(child[-1])
            
        # look at the edgelists for nodes connected to the previous chosen node and choose the one with fewest connections
        # if the chosen node has no remaining connections then choose a random unvisited node and keep going
            
        while len(child) < len(p1):
            
            candidates = [(node, len(edgelist[node])) for node in edgelist[child[-1]]]
        
            if len(candidates) > 0:        
                minedges = min([c[1] for c in candidates])
                child.append(random.choice([c[0] for c in candidates if c[1] == minedges]))
            else:
                child.append(random.choice([c for c in p1 if c not in child]))
            
            for key in edgelist.keys():
                edgelist[key].discard(child[-1])
                
        children.append(child)
        
    return children
    
# create functions to perform random mutations on the next generation
#--------------------------------------------------------------------
    
def mutateDM(children, rate):
    '''implement the displacement-mutation (DM) method of mutating a child path for genetic diversity
    a random subtour is extracted from the child and subsequently inserted in a random new location
    @params:
        - children[list]: list of child paths to potentially mutate
        - rate[float]: mutation rate (probability with which a path will be mutated)
    @return:
        - newchildren[list]: list of child paths with some children modified by mutation'''

    newchildren = []
    for child in children:
        if random.random() < rate:
            subbeg  = random.randint(0,        len(child)-1)
            subend  = random.randint(subbeg+1, len(child))
            subtour = child[subbeg:subend]
            therest = child[:subbeg] + child[subend:]
            iposit  = random.randint(0, len(child))
            newchildren.append(therest[:iposit] + subtour + therest[iposit:])
        else:
            newchildren.append(child[:])
    return newchildren
    
def mutateEM(children, rate):
    '''implement the exchange-mutation (EM) method of mutating a child path for genetic diversity
    two random elements in a path are exchanged, or in other words the positions are switched
    @params:
        - children[list]: list of child paths to potentially mutate
        - rate[float]: mutation rate (probability with which a path will be mutated)
    @return:
        - newchildren[list]: list of child paths with some children modified by mutation'''

    newchildren = []
    for child in children:
        if random.random() < rate:
            pos1 = random.randint(0, len(child)-1)
            pos2 = random.randint(0, len(child)-1)
            newchild = child[:]
            newchild[pos1], newchild[pos2] = newchild[pos2], newchild[pos1]   
            newchildren.append(newchild)
        else:
            newchildren.append(child[:])
    return newchildren
    
# set up the main genetic algorithm
#----------------------------------
    
def geneticBaseball(generations = 1000, popsize = 100, mrate = 0.05, cross = crossER, mutate = mutateDM):
    '''main implementation of the genetic algorithm to find the shortest path between all 30 MLB stadiums
    @params:
        - generations[int]: number of generations/iterations of crossover/mutation to use to produce candidate paths
        - popsize[int]: number of individuals in the population at each generation/iteration
        - mrate[float]: mutation rate to apply to children (probability that each child will be randomly mutated)
        - cross[func]: crossover/breeding function to use in the production of new children [crossOX, crossER]
        - mutate[func]: mutation function to use in the random mutation of new children [mutateDM, mutateEM]
    @return:
        - gentracker[dict]: dictionary to keep track of iterative (multiples of 100) and final results'''

    population = initPop(TEAMLIST, popsize)
    gentracker = {'generation': [], 'avgdist': [], 'mindist': [], 'maxdist': [], 'bestpos': []}
    
    for g in range(generations+1):

        if g % 100 == 0:

            print("iteration: {0}".format(g)) 
            diststats = calcDistStats(DISTDICT, population)

            gentracker['generation'].append(g)
            gentracker['avgdist'].append(round(diststats['avgdist'] / 1609.34))
            gentracker['mindist'].append(round(diststats['mindist'] / 1609.34))
            gentracker['maxdist'].append(round(diststats['maxdist'] / 1609.34))
            gentracker['bestpos'].append(",".join(population[diststats['bestpos']]))

        fitness  = calcFitness(DISTDICT, population)
        parents  = getParents(popsize, population, fitness)
        bestpath = parents[0]
        
        children = cross(popsize, parents)
        children = mutate(children, mrate)

        children.append(bestpath)
        population = children[:]     
        
    return gentracker

results = geneticBaseball(5000, 100, 0.05)
results = pd.DataFrame(results)
results.to_csv('../data/PathResults.csv', sep = ',', header = True, index = False) 

