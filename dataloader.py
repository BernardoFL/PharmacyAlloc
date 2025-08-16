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
    
    




def load_patients_from_csv_files(main_file="Data/data_table.csv", conversion_file="Data/drug_mapping_table.csv", drug_condition_file="Data/drug_condition_atc_table.csv", start_idx=None, end_idx=None):
    """
    Constructs a list of Patient objects from three CSV files.
    If start_idx and end_idx are specified, only patients within that range are loaded.
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
    grouped = list(df_main.groupby('person_id'))
    
    # Slice the grouped data if indices are provided
    if start_idx is not None and end_idx is not None:
        grouped = grouped[start_idx:end_idx]
        
    for person_id, df_patient in grouped:
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
    Constructs a JAX matrix X of shape (n_patients, n_covs, max_n_visits)
    from a list of patients with potentially variable numbers of visits.

    The second dimension (n_covs) contains the concatenation of:
      - Static patient-level continuous covariates.
      - Visit-specific binary drug indicators.

    For patients with fewer than max_n_visits, the remaining visit slots
    are padded with static covariates and zero drug indicators.

    Parameters:
        patients (List[Patient]): A list of Patient objects.

    Returns:
        jnp.ndarray: A JAX array X with shape (n_patients, n_covs, max_n_visits).
                      n_covs = num_static_covariates + num_unique_drugs.
    """
    if not patients:
        return jnp.empty((0, 0, 0))

    # --- 1. Extract Static Covariates and Find Max Visits ---
    static_cov_list = []
    max_n_visits = 0
    for i, patient in enumerate(patients):
        if not hasattr(patient, 'covariates') or not hasattr(patient, 'visits'):
             raise AttributeError(f"Patient object at index {i} missing 'covariates' or 'visits' attribute.")

        cov = np.array(patient.covariates, dtype=np.float32).flatten()
        static_cov_list.append(cov)
        max_n_visits = max(max_n_visits, len(patient.visits))

    # Handle case where no patients have any visits
    # if max_n_visits == 0:
    #     logging.warning("No visits found across all patients. Resulting matrix will have 0 visit dimension.")
        # Or raise ValueError("No visits found in any patient.") depending on desired behavior

    num_static_covs = len(static_cov_list[0]) if static_cov_list else 0

    # --- 2. Get Unique Drugs Across All Visits ---
    unique_drugs = {}
    for patient in patients:
        for visit in patient.visits:
            if "drugs" not in visit: continue
            for drug in visit["drugs"]:
                 if not hasattr(drug, 'rxcui'):
                      raise AttributeError(f"Drug object {drug} in patient {getattr(patient, 'id', i)}, visit {visit.get('visit_id', 'N/A')} missing 'rxcui' attribute.")
                 unique_drugs[str(drug.rxcui)] = True

    unique_drug_list = sorted(unique_drugs.keys())
    num_unique_drugs = len(unique_drug_list)
    drug_to_index = {rxcui: idx for idx, rxcui in enumerate(unique_drug_list)}

    n_patients = len(patients)
    n_total_covs = num_static_covs + num_unique_drugs

    # --- 3. Build the 3D Matrix with Padding ---
    all_patient_data = [] # List to hold the data for each patient (each element will be shape (n_total_covs, max_n_visits))

    # Pre-calculate the zero drug vector for padding efficiency
    zero_drug_indicator = np.zeros(num_unique_drugs, dtype=np.float32)

    for i, patient in enumerate(patients):
        static_covs = static_cov_list[i] # Shape (num_static_covs,)
        current_n_visits = len(patient.visits)

        # Pre-calculate the feature vector used for padding this patient
        padding_visit_features = np.concatenate([static_covs, zero_drug_indicator])

        patient_visit_data = [] # List to hold combined features for each visit slot (actual + padded)

        # Process actual visits
        for visit_idx, visit in enumerate(patient.visits):
            # Create drug indicator vector for THIS visit
            visit_drug_indicator = np.zeros(num_unique_drugs, dtype=np.float32)
            if "drugs" in visit:
                for drug in visit["drugs"]:
                    rxcui_str = str(drug.rxcui)
                    if rxcui_str in drug_to_index:
                        idx = drug_to_index[rxcui_str]
                        visit_drug_indicator[idx] = 1.0

            # Concatenate static covariates with the current visit's drug indicator
            combined_visit_features = np.concatenate([static_covs, visit_drug_indicator])
            patient_visit_data.append(combined_visit_features)

        # Add padding if necessary
        num_padding = max_n_visits - current_n_visits
        for _ in range(num_padding):
            patient_visit_data.append(padding_visit_features)

        # Stack the visit data for this patient: list of vectors -> 2D array
        # Resulting shape: (max_n_visits, n_total_covs)
        patient_matrix_visits_first = np.stack(patient_visit_data, axis=0)

        # Transpose to get the desired shape: (n_total_covs, max_n_visits)
        patient_matrix = patient_matrix_visits_first.T
        all_patient_data.append(patient_matrix)

    # --- 4. Stack patient matrices into the final 3D array ---
    # Input: list of (n_total_covs, max_n_visits) arrays
    # Output: shape (n_patients, n_total_covs, max_n_visits)
    final_matrix = np.stack(all_patient_data, axis=0)

    # Convert the final NumPy array to a JAX array
    return jnp.asarray(final_matrix)

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
def load_data(patient_start_idx=None, patient_end_idx=None):
    """
    Load and preprocess data, returning JAX arrays.
    Uses cached data if available and no specific patient range is requested.
    """
    cache_file = 'Data/cached/preprocessed_data.npz'
    condition_cache = 'Data/cached/condition_list.pkl'
    
    # Only use cache if no specific patient range is requested
    if patient_start_idx is None and patient_end_idx is None and os.path.exists(cache_file) and os.path.exists(condition_cache):
        cached_data = np.load(cache_file)
        A = jnp.array(cached_data['A'])
        X_cov = jnp.array(cached_data['X_cov'])
        
        with open(condition_cache, 'rb') as f:
            import pickle
            condition_list = pickle.load(f)
        
        A = 2 * A - 1
        return A, X_cov, condition_list
    
    # Load a specific slice of patients if indices are provided
    patients = load_patients_from_csv_files(start_idx=patient_start_idx, end_idx=patient_end_idx)
    
    for patient in patients:
        for visit in patient.visits:
            visit["drugs"] = [drug for drug in visit["drugs"] 
                            if hasattr(drug, "atcs") and drug.atcs is not None and len(drug.atcs) > 0]

    A, condition_to_index = construct_A_matrix(patients)
    X_cov = construct_covariate_matrix(patients)
    
    # For consistency, we should load the full condition list, not just from the shard
    full_condition_list = get_all_conditions_from_drugs(load_patients_from_csv_files())
    
    # Cache the full dataset if it was loaded
    if patient_start_idx is None and patient_end_idx is None:
        if not os.path.exists('Data/cached'):
            os.makedirs('Data/cached')
        np.savez(cache_file, A=A, X_cov=X_cov)
        with open(condition_cache, 'wb') as f:
            import pickle
            pickle.dump(full_condition_list, f)
    
    return A, X_cov, full_condition_list

