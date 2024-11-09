import matplotlib.pyplot as plt

data={'scientific_papers': 7.59, 'reddit': 19.42, 'aeslc': 16.06, 'ag_news_subset': 38.43, 'imagenet2012': 167.33, 'plant_village ': 875.5}

keys=list(data.keys())
values=list(data.values())
# Plotting the storage needs
plt.figure(figsize=(10, 6))
plt.plot(keys, values, color='skyblue')
plt.xlabel('Dataset Type')
plt.ylabel('Storage Size (Relative)')
plt.title('Storage Needs for Different Dataset Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
