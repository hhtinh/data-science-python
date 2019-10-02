import os
import geocoder
import csv
import random
import time
import pandas as pd

pd.set_option('mode.chained_assignment', None)

folder = os.path.join('data', '')
files = [file for file in os.listdir(folder) if file.endswith('.csv')]

for file in files:
    #state = file.split('_')[0]
    state = 'Vietnam'
    filename = file.split('.')[0]

    with open(os.path.join(folder, file), 'r') as f:
        df = pd.read_csv(f, names=['PERSNBR','BRANCHNBR','TAXID','ADDRNAME','latitude','longitude'], skiprows=1)
        places = df['ADDRNAME']
        count = 0

        for place in places:
            time.sleep(random.random())
            address = '{}, {}'.format(place, state)
            #print(address)
            #print(df.latitude[count])
			
            g = geocoder.arcgis(address)

            trials = 5
            for i in range(trials):
                #print(g.status)
                if g.status == 'OK':
                    df.latitude[count] = g.latlng[0]
                    df.longitude[count] = g.latlng[1]					
                    break
                elif g.status == 'ZERO_RESULTS':
                    g = geocoder.arcgis(address)
                    if i == trials - 1:
                        print("ERROR: No Result")
                else:
                    print('ERROR')
                    print(g.current_result)
                    time.sleep(1)
                    g = geocoder.arcgis(address)
            count += 1            
        #print(count)
		
    outputfile = os.path.join('data', "GEO-"+filename+".csv")
    df.to_csv(outputfile, index=False)
