import pandas as pd

# Assuming X is your features and Y is your labels
data = pd.DataFrame(X)
data['label'] = Y  # Add Y as a new column

# Save to CSV
data.to_csv('coffee_data.csv', index=False)
from IPython.display import FileLink

# Create a download link for the CSV file
FileLink('coffee_data.csv')
