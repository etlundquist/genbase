# import needed packages and set options
#---------------------------------------

import time
import requests
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.width',    100)

# create a list of tuples containing relevant info for each team
#---------------------------------------------------------------

coordinates = pd.read_csv('../data/StadiumCoordinates.csv', dtype = 'str')
print(coordinates)

coordinates = [(row.TeamCode, row.TeamName, row.Stadium, ",".join([row.Latitude, row.Longitude])) for row in coordinates.itertuples()]
print(coordinates)

# get the distance and expected driving duration between each pair of stadiums
#-----------------------------------------------------------------------------

apikey    = open('apikey.txt').read().strip()          
endpoint  = 'https://maps.googleapis.com/maps/api/distancematrix/json'
distances = {'OrigCode':[], 'OrigName':[], 'OrigAddr':[], 'DestCode':[], 'DestName':[], 'DestAddr':[], 'Distance':[], 'Duration':[]}

for i in range(30):
    
    qp = {'origins': coordinates[i][3], 'destinations': "|".join([c[3] for c in coordinates]), 'key': apikey}     
    response = requests.get(endpoint, params = qp).json()
    time.sleep(5) 

    for j in range(30):
        
        distances['OrigCode'].append(coordinates[i][0])
        distances['DestCode'].append(coordinates[j][0])
        distances['OrigName'].append(coordinates[i][2])
        distances['DestName'].append(coordinates[j][2])
        distances['OrigAddr'].append(response['origin_addresses'][0])
        distances['DestAddr'].append(response['destination_addresses'][j])      
        distances['Distance'].append(response['rows'][0]['elements'][j]['distance']['value'])
        distances['Duration'].append(response['rows'][0]['elements'][j]['duration']['value'])

# convert the results in a pandas dataframe and then output to CSV
#-----------------------------------------------------------------

df = pd.DataFrame(distances)
print("DIMENSIONS OF FINAL DATA SET: ", df.shape)
print("FREQUENCY OF ORIG VALUES: ", df.OrigCode.value_counts(), sep = "\n")
print("FREQUENCY OF DEST VALUES: ", df.DestCode.value_counts(), sep = "\n")
df.to_csv('../data/StadiumDistances.csv', sep = ',', header = True, index = False)

