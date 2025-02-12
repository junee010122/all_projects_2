import pandas as pd

# Load the generated dataset
csv_path = "output_data.csv"
data = pd.read_csv(csv_path)

# Export for Tableau
tableau_export_path = "tableau_export.csv"
data.to_csv(tableau_export_path, index=False)

print(f"Data exported for Tableau: {tableau_export_path}")

