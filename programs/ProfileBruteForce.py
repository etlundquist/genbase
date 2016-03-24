# import needed packages and set options
#---------------------------------------

import math
import random
import pandas as pd
import numpy  as np
random.seed(1492)

# bring in the distances data and create a team list and dictionary distance lookup
#----------------------------------------------------------------------------------

DISTANCES = pd.read_csv('../data/StadiumDistances.csv')
TEAMLIST  = list(set([row for row in DISTANCES.OrigCode]))
DISTDICT  = {(row.OrigCode, row.DestCode): row.Distance for row in DISTANCES.itertuples()}

# create functions to build an initial random population 
#-------------------------------------------------------

def genPath(teamlist):
    indiv    = []
    tosample = teamlist[:]
    while len(indiv) < len(teamlist):
        indiv.append(tosample.pop(random.randint(0, len(teamlist) - len(indiv) - 1)))
    return indiv

def calcDist(distdict, indiv):
    return sum([distdict[(indiv[i % len(indiv)], indiv[(i+1) % len(indiv)])] for i in range(len(indiv))])

def profileBF(iterations):
    mindist = 999999999
    for i in range(iterations):
        dist = calcDist(DISTDICT, genPath(TEAMLIST))
        if dist < mindist: mindist = dist
    return mindist
    
profileBF(100000)


