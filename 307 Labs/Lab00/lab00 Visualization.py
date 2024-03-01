import pandas as pd
import matplotlib.pyplot as plt

# Load the data
penguins_train = pd.read_csv("https://cs307.org/lab-00/data/penguins-train.csv")

# 1. Bar Chart for species counts
species_counts = penguins_train['species'].value_counts()
species_counts.plot(kind='bar')
plt.title('Penguin Species Counts')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# 2. Histogram for a continuous variable (e.g., bill depth)
penguins_train['bill_depth_mm'].hist(bins=20)
plt.title('Distribution of Bill Depth')
plt.xlabel('Bill Depth (mm)')
plt.ylabel('Frequency')
plt.show()

# 3. Box Plot for measurements across species
penguins_train.boxplot(by='species', column=['bill_depth_mm'], grid=False)
plt.title('Bill Depth by Species')
plt.xlabel('Species')
plt.ylabel('Bill Depth (mm)')
plt.suptitle('')
plt.show()

# 4. Scatter Plot for relationship between two variables (e.g., bill depth and flipper length)
plt.scatter(penguins_train['bill_depth_mm'], penguins_train['flipper_length_mm'], c=pd.factorize(penguins_train['species'])[0])
plt.title('Bill Depth vs Flipper Length by Species')
plt.xlabel('Bill Depth (mm)')
plt.ylabel('Flipper Length (mm)')
plt.colorbar(ticks=[0, 1, 2], label='Species')
plt.show()