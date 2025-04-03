import pandas as pd

# Define file paths

main_file_path = "Data/data_table.csv"
conversion_file_path = "Data/drug_mapping_table.csv"
drug_condition_file_path = "Data/drug_condition_atc_table.csv"

# ---------------------------
# Create conv_dict from conversion_file
# ---------------------------
df_conversion = pd.read_csv(conversion_file_path)
conv_dict = {}
# Assume the conversion file has at least 4 columns:
# first column is the drug name; the last three columns contain up to three rxcui codes.
for idx, row in df_conversion.iterrows():
    drug_name = row.iloc[0]
    codes = []
    for col in df_conversion.columns[-3:]:
        if pd.notna(row[col]):
            # Convert float to int then to string to remove decimals
            codes.append(str(int(row[col])))
    conv_dict[drug_name] = codes

# ---------------------------
# Create drug_dict from drug_condition_file
# ---------------------------
df_drug = pd.read_csv(drug_condition_file_path)
drug_dict = {}
# Assume df_drug has a column named 'rxcui'
for idx, row in df_drug.iterrows():
    rxcui_val = str(int(row['rxcui']))
    drug_dict[rxcui_val] = row

# ---------------------------
# Read the main file and convert covariates to dummy variables
# ---------------------------
df_main = pd.read_csv(main_file_path)

# Assume covariate columns are in positions 2 through 3 (adjust as needed)
cov_cols = df_main.columns[2:4]
df_cov = pd.get_dummies(df_main[cov_cols])
# Reassemble df_main: first two columns ('person_id', 'viscount'), then dummy covariates, then the rest.
df_main = pd.concat([df_main.iloc[:, :2], df_cov, df_main.iloc[:, 4:]], axis=1)

# ---------------------------
# Identify medication columns and determine which to delete
# ---------------------------
# Medication columns are those after the first two columns + dummy covariate columns.
med_start = 2 + len(df_cov.columns)
cols_to_delete = []

for col in df_main.columns[med_start:]:
    # The column header is the drug name.
    drug_name = col
    # Check if the drug name is not in conv_dict.
    if drug_name not in conv_dict:
        cols_to_delete.append(col)
    else:
        rxcui_list = conv_dict[drug_name]
        # Check if at least one of the rxcui codes exists in drug_dict.
        if not any(str(rxcui) in drug_dict for rxcui in rxcui_list):
            cols_to_delete.append(col)

# ---------------------------
# Delete identified medication columns and overwrite the file
# ---------------------------
df_main.drop(columns=cols_to_delete, inplace=True)
df_main.to_csv(main_file_path, index=False)

print(f"Deleted columns: {cols_to_delete}")