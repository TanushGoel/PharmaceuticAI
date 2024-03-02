from chembl_webresource_client.new_client import new_client
import pandas as pd

class Client():

    def search(self): # search for target proteins --> returns its inhibitors and its FASTA
        target = str(input("Target:"))
        print("Searching for", target+"...")
        query = new_client.target.search(target)
        results = pd.DataFrame.from_dict(query)
        if len(results) <= 1:
            print("No results found")
            return None, None
        results.drop_duplicates(subset="pref_name", inplace=True)
        results = results[results['target_components'].apply(lambda x: len(x) > 0)]
        results = results.reset_index(drop=True)
        display(results)
        
        target = int(input("Which Protein From Above? (#):"))
        chembl_id = results["target_chembl_id"].iloc[target]
        
        print("Retrieving FASTA for", results["pref_name"].iloc[target]+"...")
        target_comps = new_client.target.get(chembl_id)["target_components"]
        comp_ids = target_comps[0]["component_id"]
        fasta = new_client.target_component.get(comp_ids)["sequence"]
        
        print("Finding Inhibitors for", results["pref_name"].iloc[target]+"...")    
        inhibitors = new_client.activity.filter(target_chembl_id=chembl_id).filter(standard_type="Inhibition")
        inhibitors = pd.DataFrame.from_dict(inhibitors)
        if len(inhibitors):
            return inhibitors["canonical_smiles"].unique().tolist(), fasta
        return None, fasta