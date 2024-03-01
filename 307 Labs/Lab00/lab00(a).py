import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

penguins_train = pd.read_csv("https://cs307.org/lab-00/data/penguins-train.csv")

species_counts = penguins_train['species'].value_counts()

# Calculate the proportion of each species
total_count = len(penguins_train) ## shouldn't this be -1?
species_proportion = species_counts / total_count

# Compute the mean and standard deviation of bill depth for each species
bill_depth_stats = penguins_train.groupby('species')['bill_depth_mm'].agg(['mean', 'std'])
bill_length_stats = penguins_train.groupby('species')['bill_length_mm'].agg(['mean', 'std'])

# Display the results
print("Species Counts:\n", species_counts)
print("\nSpecies Proportion:\n", species_proportion)
print("\nBill Depth Stats (mean and std):\n", bill_depth_stats)
print("\nBill Length Stats (mean and std):\n", bill_length_stats)