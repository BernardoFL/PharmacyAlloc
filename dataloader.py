import jax
import jax.numpy as jnp
import pandas as pd
from typing import List, Dict, Any, Tuple
import os
import numpy as np

class Drug:
    """
    Represents a drug with an rxcui and associated ATC classifications.
    
    Attributes:
        rxcui (str): The rxcui identifier.
        atcs (List[jnp.ndarray]): A list of JAX arrays, each representing an ATC classification.
            Each ATC classification is encoded as an array of shape (5,) of integer codes.
        Conditions (List[str]): A list of condition names (strings) this drug is used for.
    """
    # Class-level vocabulary for ATC level encoding.
    atc_vocab = {}
    next_atc_code = 1

    @staticmethod
    def encode_atc_level(level):
        if level in Drug.atc_vocab:
            return Drug.atc_vocab[level]
        else:
            code = Drug.next_atc_code
            Drug.atc_vocab[level] = code
            Drug.next_atc_code += 1
            return code

    @staticmethod
    def build_atc_tensor(atc_levels):
        codes = [Drug.encode_atc_level(level) for level in atc_levels]
        return jnp.array(codes, dtype=jnp.int32)
    
    def __init__(self, rxcui, atc_codes, conditions=None):
        """
        :param rxcui: The rxcui identifier (string).
        :param atc_codes: A list of ATC classifications, where each classification is a list of 4 descriptive strings.
        :param conditions: (Optional) A list of condition names (strings) this drug is used for.
        """
        self.rxcui = rxcui
        if len(atc_codes) > 20:
            raise ValueError("A drug can have at most 20 ATC classifications.")
        
        self.atcs = []
        for code in atc_codes:
            if len(code) != 4:
                raise ValueError("Each ATC classification must consist of 4 levels.")
            atc_tensor = Drug.build_atc_tensor(code)
            self.atcs.append(atc_tensor)
        
        self.Conditions = conditions if conditions is not None else []
    
    def get_atc_codes(self):
        return self.atcs
    
    def __repr__(self):
        atcs_repr = [atc.tolist() for atc in self.atcs]
        return f"Drug(rxcui={self.rxcui}, atcs={atcs_repr}, Conditions={self.Conditions})"

class Condition:
    """
    Represents a medical condition associated with a set of drugs.
    
    Attributes:
        name (str): The name of the condition.
        drugs (List[Drug]): A list of Drug objects that are associated with this condition.
    """
    def __init__(self, name, drugs=None):
        self.name = name
        self.drugs = drugs if drugs is not None else []
    
    def add_drug(self, drug):
        """Add a drug to this condition if not already present."""
        if drug not in self.drugs:
            self.drugs.append(drug)
    
    def __repr__(self):
        drug_ids = [drug.rxcui for drug in self.drugs]
        return f"Condition(name={self.name}, drugs={drug_ids})"


class Patient:
    """
    Class representing a patient with multiple visits.
    
    Attributes:
        person_id (str): A unique identifier for the patient.
        visits (List[dict]): A list of dictionaries, each representing one visit.
            Each visit dictionary has:
                - 'viscount': The visit identifier,
                - 'covariates': A JAX array of numeric covariates for that visit,
                - 'drugs': A list of Drug objects representing the drugs taken during that visit.
        conditions (List[str]): A list of conditions for the patient (variable length).
        covariates (jnp.ndarray): A patient-level covariate array. If not provided,
            it is computed as the mean of the visit-level covariates.
    """
    def __init__(self, person_id, visits, conditions=None, covariates=None):
        """
        :param person_id: Unique patient identifier.
        :param visits: A list of visit dictionaries. Each dictionary should contain:
                       - 'viscount': a visit identifier,
                       - 'covariates': the covariate data for that visit (as a list or JAX array),
                       - 'drugs': a list of Drug objects.
        :param conditions: (Optional) A list of Condition objects for the patient.
        :param covariates: (Optional) A patient-level covariate array. If not provided, it is computed as the mean
                           of the visit-level covariates.
        """
        self.person_id = person_id
        self.visits = []
        for visit in visits:
            # Ensure that visit covariates are stored as JAX array.
            covs = visit["covariates"]
            if not isinstance(covs, jnp.ndarray):
                covs = jnp.array(covs, dtype=jnp.float32)
            visit["covariates"] = covs
            self.visits.append(visit)
        
        # Now conditions is expected to be a list of Condition objects.
        self.conditions = conditions if conditions is not None else []
        
        if covariates is not None:
            self.covariates = covariates if isinstance(covariates, jnp.ndarray) else jnp.array(covariates, dtype=jnp.float32)
        else:
            if self.visits:
                # Stack visit-level covariates and take the mean along visits.
                all_covs = jnp.stack([visit["covariates"] for visit in self.visits])
                self.covariates = jnp.mean(all_covs, axis=0)
            else:
                self.covariates = jnp.array([], dtype=jnp.float32)


    def get_fused_atc_graph(self, t):
        """
        Returns a fused ATC graph for the visit at index t.
        This method fuses the ATC classifications of all drugs taken in visit t into a single directed graph.
        Each ATC classification is stored as a JAX array of shape (5,) containing integer codes.
        Nodes in the fused graph are represented as (level, code) tuples (with level from 1 to 5),
        and an edge is added from (level, code) to (level+1, next_code) for each consecutive pair in the ATC sequence.
        
        Parameters:
            t (int): The visit index (time) for which to obtain the fused ATC graph.
        Returns:
            A networkx.DiGraph representing the fused ATC graph.
        """
        fused_graph = nx.DiGraph()
        if t < 0 or t >= len(self.visits):
            print(f"Invalid visit index {t} for patient {self.person_id}")
            return fused_graph
        
        drugs = self.visits[t].get("drugs", [])
        for drug in drugs:
            for atc_tensor in drug.atcs:
                atc_list = atc_tensor.tolist()  # This is a list of 5 integer codes.
                # Add nodes for each level in the ATC classification.
                for level, code in enumerate(atc_list, start=1):
                    fused_graph.add_node((level, code))
                # Add edges between consecutive levels.
                for level in range(1, len(atc_list)):
                    fused_graph.add_edge((level, atc_list[level-1]), (level+1, atc_list[level]))
        return fused_graph    
    
    def __repr__(self):
        visits_repr = []
        for visit in self.visits:
            visits_repr.append({
                "viscount": visit["viscount"],
                "covariates": visit["covariates"].tolist(),
                "drugs": [repr(drug) for drug in visit["drugs"]]
            })
        return f"Patient(person_id={self.person_id}, visits={visits_repr}, conditions={self.conditions}, covariates={self.covariates.tolist()})"
    
    




def load_patients_from_csv_files(main_file="Data/data_table.csv", conversion_file="Data/drug_mapping_table.csv", drug_condition_file="Data/drug_condition_atc_table.csv"):
    """
    Constructs a list of Patient objects from three CSV files.
    
    Parameters:
        main_file: CSV file with patient visit data. Columns include 'person_id', 'viscount', covariate columns,
                   and medication columns (with drug names as headers and binary indicators as values).
        conversion_file: CSV file mapping drug names (as in main_file) to rxcui codes.
                         Expected: first column is drug name; last three columns contain up to three rxcui codes.
        drug_condition_file: CSV file (drug_condition_atc_table.csv) with drug info.
                             Expected columns: 'rxcui', 'standard', and for i=1,...,20:
                              - 'Condition-i',
                              - 'ATC-i-level-1', 'ATC-i-level-2', 'ATC-i-level-3', 'ATC-i-level-4'
    
    Returns:
        A list of Patient objects. For each patient, the visits are constructed from the CSV.
        Each visit contains a list of Drug objects.
        Each Drug object is created by reading its ATC classifications and conditions from the drug_condition_file.
        Additionally, each patient’s conditions attribute is a list of Condition objects constructed from the drugs taken.
    """
    # Read CSV files.
    df_main = pd.read_csv(main_file)
    df_conversion = pd.read_csv(conversion_file)
    df_drug = pd.read_csv(drug_condition_file)
    
    # Optional: convert covariate columns to dummy variables.
    # Here we assume covariate columns are in positions 2 through 3.
    cov_cols = df_main.columns[2:4]
    df_cov = df_main[cov_cols]
    
    # Create mapping from drug name to list of rxcui codes.
    conv_dict = {}
    for idx, row in df_conversion.iterrows():
        drug_name = row.iloc[0]
        codes = []
        for col in df_conversion.columns[-3:]:
            if pd.notna(row[col]):
                codes.append(str(int(row[col])))
        conv_dict[drug_name] = codes
    
    # Create mapping from rxcui (as string) to the corresponding row in df_drug.
    drug_dict = {}
    for idx, row in df_drug.iterrows():
        rxcui_val = str(int(row['rxcui']))
        drug_dict[rxcui_val] = row
    
    patients = []
    
    # Group rows by person_id.
    for person_id, df_patient in df_main.groupby('person_id'):
        visits = []
        # We'll collect conditions for this patient in a dictionary mapping condition name to Condition object.
        patient_conditions_dict = {}
        # Sort visits by viscount.
        df_patient_sorted = df_patient.sort_values(by='viscount')
        
        # Use covariate columns from df_cov.
        covariate_columns = list(df_cov.columns)
        # Medication columns: assume they start after the first 2 + len(covariate_columns) columns.
        med_columns = df_main.columns[2 + len(covariate_columns):]
        
        for _, row in df_patient_sorted.iterrows():
            viscount = row['viscount']
            covariates = row[covariate_columns].tolist()
            drugs = []
            # Process each medication column.
            for med_col in med_columns:
                # The column header is the drug name; process only if the cell value is 1.
                if pd.isna(row[med_col]) or row[med_col] != 1:
                    continue
                med_name = med_col
                if med_name not in conv_dict:
                    print(f"Warning: no rxcui codes found for drug name: {med_name}")
                    continue
                rxcui_list = conv_dict[med_name]
                for rxcui in rxcui_list:
                    rxcui_str = str(rxcui)
                    if rxcui_str not in drug_dict:
                        continue
                    drug_row = drug_dict[rxcui_str]
                    atc_codes = []
                    # For i from 1 to 20, check for an ATC classification (with 4 levels).
                    for i in range(1, 21):
                        atc_key = f"ATC-{i}-level-1"
                        if atc_key in drug_row and pd.notna(drug_row[atc_key]):
                            atc = []
                            valid = True
                            for k in range(1, 5):  # Expecting 4 levels.
                                level_key = f"ATC-{i}-level-{k}"
                                if level_key in drug_row and pd.notna(drug_row[level_key]):
                                    atc.append(str(drug_row[level_key]))
                                else:
                                    valid = False
                                    break
                            if valid and len(atc) == 4:
                                atc_codes.append(atc)
                        else:
                            break
                    # Gather conditions for this drug.
                    conditions = []
                    for i in range(1, 21):
                        cond_key = f"Condition-{i}"
                        if cond_key in drug_row and pd.notna(drug_row[cond_key]):
                            conditions.append(str(drug_row[cond_key]))
                    # Create a Drug object with its conditions.
                    drug_obj = Drug(rxcui=rxcui_str, atc_codes=atc_codes, conditions=conditions)
                    drugs.append(drug_obj)
                    # If this drug has exactly one condition, use it to update the patient's condition dictionary.
                    if len(conditions) == 1:
                        cond_name = conditions[0]
                        if cond_name not in patient_conditions_dict:
                            # Create a new Condition object.
                            patient_conditions_dict[cond_name] = Condition(cond_name, drugs=[drug_obj])
                        else:
                            patient_conditions_dict[cond_name].add_drug(drug_obj)
            visit = {
                "viscount": viscount,
                "covariates": covariates,
                "drugs": drugs
            }
            visits.append(visit)
        # Create a Patient object with the list of Condition objects.
        patient_obj = Patient(person_id=person_id, visits=visits, conditions=list(patient_conditions_dict.values()))
        patients.append(patient_obj)
    
    return patients
# Example call:
# patients = load_patients_from_csv_files("main.csv", "conversion.csv", "drug_condition_atc_table.csv")

def construct_A_matrix(patients: List[Patient]) -> Tuple[jnp.ndarray, Dict[str, int]]:
    """
    Constructs a 3D JAX array A such that:
      A[i, j, t] == 1 if, at visit t, patient i takes condition j,
      and 0 otherwise.
      
    A patient is considered to take condition j at a visit if one of the drugs they take
    in that visit has a Conditions list of length 1 and that single element equals j.
    
    Parameters:
        patients (list): A list of Patient objects. Each Patient should have:
                  - visits: a list of visit dictionaries, where each visit has a key "drugs"
                    that holds a list of Drug objects. Each Drug object has an attribute "Conditions"
                    (a list of strings).
    
    Returns:
        A: A JAX array of shape (N, num_conditions, T_max), where:
           N is the number of unique patients (by person_id),
           num_conditions is the number of unique conditions (from drugs with a single condition),
           T_max is the maximum number of visits any patient has.
        condition_to_index: A dictionary mapping each unique condition name to its column index in A.
    """

    # First, collect all unique conditions from drugs that have exactly one condition.
    condition_set = set()
    for patient in patients:
        for visit in patient.visits:
            for drug in visit["drugs"]:
                if len(drug.Conditions) == 1:
                    condition_set.add(drug.Conditions[0])
    unique_conditions = sorted(condition_set)
    num_conditions = len(unique_conditions)
    
    # Build a mapping from condition name to a column index.
    condition_to_index = {cond: idx for idx, cond in enumerate(unique_conditions)}
    
    # Deduplicate patients by person_id.
    unique_patients_dict = {patient.person_id: patient for patient in patients}
    unique_patients = sorted(unique_patients_dict.values(), key=lambda p: p.person_id)
    N = len(unique_patients)
    
    # Determine the maximum number of visits across unique patients.
    T_max = max(len(patient.visits) for patient in unique_patients)
    
    # Initialize the A array with zeros.
    A = jnp.zeros((N, num_conditions, T_max), dtype=jnp.int32)
    
    # Fill the matrix:
    # For each unique patient, for each visit, and for each drug taken in that visit:
    # if the drug's Conditions list has exactly one element, mark that condition as taken.
    for i, patient in enumerate(unique_patients):
        for t, visit in enumerate(patient.visits):
            for drug in visit["drugs"]:
                if len(drug.Conditions) == 1:
                    cond = drug.Conditions[0]
                    if cond in condition_to_index:
                        j = condition_to_index[cond]
                        A = A.at[i, j, t].set(1)
                        
    return A, condition_to_index

def construct_covariate_matrix(patients: List[Patient]) -> jnp.ndarray:
    """
    Constructs a JAX matrix X of shape (N, d_total) from a list of patients,
    where N is the number of patients and d_total is the sum of:
      - the number of patient-level covariates (continuous) and
      - the number of unique drugs (binary indicators indicating if the patient ever took that drug).
      
    The original patient covariates are taken from the 'covariates' attribute.
    The drug indicators are computed by examining each patient's visits and drugs.
    
    Parameters:
        patients (list): A list of Patient objects.
        
    Returns:
        jnp.ndarray: A JAX array X with shape (N, d_total) where each row contains the concatenated
                      patient-level covariates and drug indicator vector.
    """

    # Extract the patient-level covariate array from each patient.
    # Each patient.covariates is assumed to be a 1D array (or can be flattened to one).
    cov_list = [patient.covariates.flatten() for patient in patients]

    # Compute a list of all unique drugs (by rxcui) across all patients.
    # We assume each Drug object has a unique rxcui (a string).
    unique_drugs = {}
    for patient in patients:
        for visit in patient.visits:
            for drug in visit["drugs"]:
                unique_drugs[drug.rxcui] = True
    unique_drug_list = sorted(unique_drugs.keys())
    num_drugs = len(unique_drug_list)
    
    # Build a mapping from drug rxcui to a column index.
    drug_to_index = {rxcui: idx for idx, rxcui in enumerate(unique_drug_list)}
    
    # For each patient, create a binary indicator vector of length num_drugs.
    drug_indicators = []
    for patient in patients:
        indicator = jnp.zeros(num_drugs)
        for visit in patient.visits:
            for drug in visit["drugs"]:
                # Mark 1 if the patient has taken this drug at least once.
                idx = drug_to_index[drug.rxcui]
                indicator = indicator.at[idx].set(1)
        drug_indicators.append(indicator)
    
    # Concatenate each patient's original covariates with their drug indicator vector.
    augmented_features = []
    for cov, indicator in zip(cov_list, drug_indicators):
        augmented = jnp.concatenate([cov, indicator])
        augmented_features.append(augmented)
    
    # Stack into a matrix: shape (N, d_total)
    return jnp.stack(augmented_features, axis=0)

def get_all_drugs(list_of_patients):
    """
    Gets a set of all unique drugs (by rxcui) from a list of patients.
        
    Parameters:
        list_of_patients: A list of Patient objects.
            
    Returns:
        set: A set of unique Drugs.
    """
    unique_drugs = []
    for patient in list_of_patients:
        for visit in patient.visits:
            for drug in visit["drugs"]:
                unique_drugs.append(drug)
    return unique_drugs

def get_all_conditions_from_drugs(patients):
    """
    Extracts and returns a sorted list of unique Condition objects from the drugs that patients take.

    For each patient, each visit, and each drug in that visit, the function examines the drug's Conditions attribute.
    If a condition (a string) is found, a Condition object is created (or updated) such that the associated drug is added
    to that condition. The result is a sorted list (by condition name) of unique Condition objects.

    Parameters:
        patients (list): A list of Patient objects. Each Patient has visits, and each visit contains a list of Drug objects.
                         Each Drug object has an attribute `Conditions`, which is a list of condition names (strings).

    Returns:
        List[Condition]: A sorted list of unique Condition objects.
    """
    condition_dict = {}
    for patient in patients:
        for visit in patient.visits:
            for drug in visit["drugs"]:
                for cond in drug.Conditions:
                    if cond not in condition_dict:
                        condition_dict[cond] = Condition(cond, drugs=[drug])
                    else:
                        # Only add the drug if it's not already present in this condition.
                        if drug not in condition_dict[cond].drugs:
                            condition_dict[cond].add_drug(drug)
    # Sort the conditions by name.
    condition_list = list(condition_dict.values())
    condition_list.sort(key=lambda c: c.name)
    return condition_list

# Example usage:
# conditions = get_all_conditions_from_drugs(patients)
# print("Conditions from drugs:", conditions)
def load_data():
    """
    Load and preprocess data, returning JAX arrays.
    Uses cached data if available.
    """
    # Check for cached data
    cache_file = 'Data/cached/preprocessed_data.npz'
    condition_cache = 'Data/cached/condition_list.pkl'
    
    if os.path.exists(cache_file) and os.path.exists(condition_cache):
        # Load from cache
        cached_data = np.load(cache_file)
        A = jnp.array(cached_data['A'])
        X_cov = jnp.array(cached_data['X_cov'])
        
        with open(condition_cache, 'rb') as f:
            import pickle
            condition_list = pickle.load(f)
        
        return A, X_cov, condition_list
    
    # If no cache, load and process as before
    patients = load_patients_from_csv_files()
    #remove patients with no drugs
    for patient in patients:
        for visit in patient.visits:
            # Reassign the drugs list to only those drugs that have a non-empty atcs attribute.
            visit["drugs"] = [drug for drug in visit["drugs"] 
                            if hasattr(drug, "atcs") and drug.atcs is not None and len(drug.atcs) > 0]

    A, condition_to_index = construct_A_matrix(patients)
    X_cov = construct_covariate_matrix(patients)
    condition_list = get_all_conditions_from_drugs(patients)
    
    # Save to cache
    np.savez(cache_file, A=A, X_cov=X_cov)
    with open(condition_cache, 'wb') as f:
        import pickle
        pickle.dump(condition_list, f)
    
    return A, X_cov, condition_list

