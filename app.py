import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, k_means

df = pd.read_csv('final_stars.csv')
print(df.head())

# df.drop(['Unnamed: 0'], axis=1, inplace= True)
# df["Mass"] = df["Mass"].apply(lambda x:x.replace('$', '').replace(',','')).astype('float')

# df['Mass'] = 1.989e+30*df['Mass']
# print(df.head())
# df['Radius'] = 6.957e+8*df['Radius']

# radius = df['Radius'].to_list() 
# mass = df['Mass'].to_list() 
# gravity =[] 
# def convert_to_si(radius,mass): 
#     for i in range(0,len(radius)-1): radius[i] = radius[i]*6.957e+8 
#     mass[i] = mass[i]*1.989e+30 

# convert_to_si(radius,mass)

# G = 6.674e-11

# for i in range(0,len(mass)):
#     g = (mass[i]*G)/((radius[i])**2)
#     gravity.append(g)

# print(gravity)

# df["gravity"] = gravity

mass = []
radius = []

for planet_data in df:
    mass.append(planet_data[3])
    radius.append(planet_data[4])

mass.sort()
radius.sort()

X = []
for index, planet_mass in enumerate(mass):
    temp_list = [radius[index], planet_mass]
    X.append(temp_list)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), wcss, markers="0", color = "red")
plt.title('Mass vs Radius') 
plt.xlabel('Mass') 
plt.ylabel('Radius') 
plt.show() 




