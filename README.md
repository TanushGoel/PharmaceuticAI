# PharmaceuticAI
Drug discovery is a very long and expensive process. The average time from FDA application to approval of drugs is 12 years, and the estimated average cost of taking a new drug from concept to market exceeds $1 billion. Of up to 10,000 compounds tested, only one may end up becoming a drug that reaches the market. 

PharmaceuticAI was developed to help make this process more time/cost effective, via an iterative process that uses multiple models to generate the best inhibitor for a given target protein. 

## [Project Description Document PDF](https://github.com/TanushGoel/PharmaceuticAI/blob/master/PharmaceuticAI.pdf)

### Data
Drug-like Compounds Dataset - [ChEMBL](https://www.ebi.ac.uk/chembl/) (a manually curated database of bioactive molecules with drug-like properties)

IC50 Dataset - Davis Dataset and KIBA Dataset

### PharmaceuticAI_CuDNNLSTM.ipynb
This notebook takes advantage of NVIDIA's CUDA Deep Neural Network library (cuDNN) (a GPU-accelerated library for deep neural networks) to train a CuDNNLSTM model able to predict the next element of a compound in SMILES format (the end being represented through the use of a "$" token). This model can then generate new compounds by creating completely random compounds, generating using certain compounds as input, or augmenting existing compounds (the augment function also has a parameter for the minimum similarity between the augmented and original compound, the number of changes made to each compound, as well as the amount of times it tries to make a valid molecule). Each compound is a valid molecule that follows the laws of chemistry and SMILES formatting, follows the 4 criteria of Lipinski's Rule of Five, and that is not already in the list of generated compounds. 

### PharmaceuticAI_Affinity.ipynb
In this notebook, I trained 1-D convolutional neural network model that is able to predict the binding affinity of a ligand to a protein given the ligand in SMILES format and the protein FASTA (amino acid sequence). 

### PharmaceuticAI_Pre-Trained_Use.ipynb
This notebook acts as the frontend for PharmaceuticAI. After uploading the best versions of the models and the drug-like compounds dataset, the models will create the best drug candidate for the inputted target protein.

These compounds can then make their way through the next stages of drug development (eg. clinical trials)
