#!/usr/bin/env python
# coding: utf-8

# # Capstone  - Week 5 - Final Project 
# # Battle of Buffalo Neighborhoods

# First off, we will install and import the packages we will need

# In[77]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

get_ipython().system('conda install -c conda-forge lxml --yes')
import lxml
import html5lib

print('Libraries imported.')


# In[78]:


import matplotlib.pyplot as plt


# ### Now , let's read in the data that has the Buffalo neighborhood names and their lat long values .

# In[79]:


df = pd.read_csv("Buffalo Neighborhood Metrics.csv")

df.head()


# Now, let's select only the neighborhood names and the lat long values.

# In[80]:


df = df[["Neighborhood","Latitude","Longitude" ]]
df.head()


# In[81]:


df.shape


# So we have 35 neighborhoods.

# In[82]:


address = 'Buffalo, NY'

geolocator = Nominatim(user_agent="buffalo_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of Buffalo are {}, {}.'.format(latitude, longitude))


# ## Explore Buffalo

# Let's create a map of Buffalo with neighborhoods superimposed on top.

# In[83]:


map_buffalo = folium.Map(location=[latitude, longitude], zoom_start= 10)

#add markers to map
for lat, lon, neighborhood in zip (df["Latitude"], df["Longitude"], df["Neighborhood"]):
  label = '{}'.format(neighborhood)
  label = folium.Popup(label, parse_html=True)
  folium.CircleMarker(
      [lat,lon],
      radius = 5,
      popup = label,
      color= "blue",
      fill=True,
      fill_color = "3186cc",
      fill_opacity=0.7,
      parse_html=False  ).add_to(map_buffalo)
map_buffalo


# # Defining Foursquare credentials

# In[84]:


CLIENT_ID = '2HOQYJJ21GVO03ILUUGJGT3OYI5ISPLUW55K31WE1NU3TOHY' # my Foursquare ID
CLIENT_SECRET = 'SV2XROTJRPQDZOBEHK30PYUXFVVL3TSSDZZE4JIL2KGHGI1M' # my Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('My credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# # Let's explore the first neighborhood in our dataframe.

# Get the neighborhood's name.

# In[85]:


neighborhood_name = df.loc[0,"Neighborhood"]
neighborhood_latitude = df.loc[0,"Latitude"]
neighborhood_longitude = df.loc[0,"Longitude"]
print(neighborhood_name)
print(neighborhood_latitude)
print(neighborhood_longitude)


# ## Now, let's get the top 100 venues that are in Central neighborhood within a radius of 500 meters

# In[86]:



LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius

 # create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[87]:


results = requests.get(url).json()
results


# In[88]:


#function that extracts the category of the venue
def get_category_type(row):
  try:
    categories_list = row['categories']
  except:
    categories_list=row['venue.categories']
  
  if len(categories_list)==0:
    return None
  else:
    return categories_list[0]['name']


# Now we are ready to clean the json and structure it into a dataframe

# In[89]:



venues=results['response']['groups'][0]['items']
nearby_venues = pd.json_normalize(venues) # this will flatten the json
 
#filtered columns
filtered_columns = ['venue.name','venue.categories','venue.location.lat','venue.location.lng']
nearby_venues=nearby_venues.loc[:,filtered_columns]
nearby_venues = nearby_venues.reindex(columns = filtered_columns)

#filter the category for each row
nearby_venues['venue.categories']=nearby_venues.apply(get_category_type, axis=1)

#clean column names
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# In[90]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# # 2. Explore Neighborhoods in Buffalo

# Let's create a function to repeat the same process to all the neighborhoods in Buffalo

# In[91]:


#the function
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[92]:


# type your answer here
buffalo_venues = getNearbyVenues(names=df['Neighborhood'],
                                   latitudes=df['Latitude'],
                                   longitudes=df['Longitude']
                                  )


# #### Let's check the size of the resulting dataframe

# In[93]:


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


# In[94]:


print(buffalo_venues.shape)
buffalo_venues


# In[95]:


buffalo_venues.groupby('Neighborhood').count().sort_values("Venue", ascending = False)


# Let's find out how many unique categories can be curated from all the returned venues

# In[96]:


print('There are {} uniques categories.'.format(len(buffalo_venues['Venue Category'].unique())))


# In[97]:


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)


# In[98]:


buffalo_venues.groupby("Venue Category").count().sort_values("Venue",ascending=False)


# # 3. Analyze Each Neighborhood

# In[99]:


# one hot encoding
buffalo_onehot = pd.get_dummies(buffalo_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
buffalo_onehot['Neighborhood']= buffalo_venues['Neighborhood'] 

buffalo_onehot.head(20)


# And let's examine the new dataframe size.

# In[100]:


# move neighborhood column to the first column
fixed_columns = [buffalo_onehot.columns[-1]] +  list(buffalo_onehot.columns[:-1])

buffalo_onehot = buffalo_onehot[fixed_columns]

buffalo_onehot.head()


# In[101]:


buffalo_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
# 

# In[102]:


buffalo_grouped = buffalo_onehot.groupby('Neighborhood').mean().reset_index()

buffalo_grouped.head(15)


# In[103]:


buffalo_grouped.shape


# ### Let's print each neighborhood along with the top 5 most common venues

# In[104]:


num_top_venues = 5

for hood in buffalo_grouped['Neighborhood']:
    print("===== "+hood+" =====")
    temp = buffalo_grouped[buffalo_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a _pandas_ dataframe

# First, let's write a function to sort the venues in descending order.

# In[105]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[106]:


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 40)


# In[107]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = buffalo_grouped['Neighborhood']

for ind in np.arange(buffalo_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(buffalo_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# In[108]:


neighborhoods_venues_sorted.shape


# As can be seen in the table above, the most common venue int the West Side neighborhood is Vietnamese restaurants.
# This means, most probably there is a big Vietnamese coommunity living in and around that neighborhood. 

# ## Types of Service Calls From West Side

# Let's see what type of service calls were made from West Side using the service calls dataset

# In[109]:


#Let's read in the data
calls = pd.read_csv("311_Service_Requests.csv")
#drop NAs

print(calls.shape)
calls


# In[110]:


#drop NAs
calls = calls.dropna()


# In[111]:


calls


# Now e have 52033 calls left to work on. Let's see how many call types per neighborhood and then look at the West Side data separately.

# In[112]:


calls_grouped = pd.DataFrame(calls[["NEIGHBORHOOD","TYPE"]].groupby(["NEIGHBORHOOD","TYPE"]).size())
 #calls[["TYPE","NEIGHBORHOOD"]].groupby(["TYPE","NEIGHBORHOOD"]).count().sort_values("NEIGHBORHOOD", ascending=False).head(20)
calls_grouped


# Now let's look at the calls that were received from West Side

# In[113]:


West_Side_Calls = calls[calls["NEIGHBORHOOD"]=="West Side"]


# In[114]:


West_Side_Calls[["TYPE", "NEIGHBORHOOD"]].groupby("TYPE").count().sort_values("NEIGHBORHOOD", ascending=False)


# So only 8 calls were received from West Side. Great!

# ## 4. Cluster Neighborhoods

# Run _k_-means to cluster the neighborhood into 5 clusters.

# In[115]:


# set number of clusters
kclusters = 4

buffalo_grouped_clustering = buffalo_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(buffalo_grouped_clustering)


# In[116]:


# check cluster labels generated for each row in the dataframe
#kmeans.labels_[0:10]
kmeans.labels_[0:40] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
# 

# In[117]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

buffalo_merged = df
#merge buffalo_grouped with buffalo_data to add latitude/longitude for each neighborhood
buffalo_merged = buffalo_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



# In[118]:


buffalo_merged.head(40)


# In[119]:


buffalo_merged = buffalo_merged.dropna()
# check the last columns!


# In[120]:


buffalo_merged.shape


# Finally, let's visualize the resulting clusters

# In[121]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(buffalo_merged['Latitude'], buffalo_merged['Longitude'], buffalo_merged['Neighborhood'], buffalo_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## 5. Examine Clusters

# Now, we can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, we can then assign a name to each cluster.

# ### Cluster 1

# In[122]:


cluster1 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 0, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster1.shape)
cluster1


# ***Cluster 2***

# In[123]:


cluster2 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 1, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster2.shape)
cluster2


# ***Cluster 3***

# In[124]:


cluster3 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 2, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster3.shape)
cluster3


# ***Cluster 4***

# In[125]:


cluster4 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 3, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster4.shape)
cluster4


# ***Cluster 5***

# In[126]:


cluster5 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 4, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster5.shape)
cluster5


# ***Cluster 6***

# In[127]:


cluster6 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 5, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster6.shape)
cluster6


# ***Cluster 7***

# In[128]:


cluster7 = buffalo_merged.loc[buffalo_merged['Cluster Labels'] == 6, buffalo_merged.columns[[0] + list(range(4, buffalo_merged.shape[1]))]]
print(cluster7.shape)
cluster7


# ## Done!!
