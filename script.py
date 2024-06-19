import os
import pandas as pd
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Function to read SMILES file
def read_smiles_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [(line.split(None, 1)[0], line.split(None, 1)[1].strip()) for line in lines]

# Function to calculate fingerprints
def calculate_fingerprints(mol, fingerprint_type):
    if fingerprint_type == 'Morgan':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    elif fingerprint_type == 'RDKit':
        return Chem.RDKFingerprint(mol)
    elif fingerprint_type == 'MACCS':
        return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)

# Function to calculate molecular properties
def calculate_molecular_properties(mol):
    properties = {}
    for desc in Descriptors.descList:
        try:
            properties[desc[0]] = desc[1](mol)
        except:
            properties[desc[0]] = None
    return properties

# Function to calculate SiRMS descriptors
def calculate_sirms_descriptors(smiles):
    result = subprocess.run(["sirms", "-i", smiles], capture_output=True, text=True)
    output = result.stdout.strip().split('\n')
    descriptors = {}
    for line in output:
        if ':' in line:
            key, value = line.split(':')
            descriptors[key.strip()] = float(value.strip())
    return descriptors

# Function to calculate QED values
def calculate_qed(mol):
    qed = Chem.QED.default(mol)
    return {"QED": qed}

# Function to convert SMILES to SDF using RDKit
def convert_smiles_to_sdf(smiles_file, output_sdf_file):
    suppl = Chem.SmilesMolSupplier(smiles_file, titleLine=False)
    writer = Chem.SDWriter(output_sdf_file)
    for mol in suppl:
        if mol is not None:
            mol = Chem.AddHs(mol)  # Add hydrogens for 3D conversion
            AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
            AllChem.MMFFOptimizeMolecule(mol)  # Optimize 3D geometry
            writer.write(mol)
    writer.close()

# Function to prompt user for option
def prompt_for_option(options, option_type):
    print(f"\nSelect which {option_type} do you want to use for descriptor calculation:")
    for i, option in enumerate(options, 1):
        print(f"{i}) {option}")
    choice = int(input("Enter the number corresponding to your choice: "))
    return options[choice - 1], choice

# Function to prompt user to choose file
def prompt_for_file(files, file_type):
    print(f"\nSelect the {file_type} file:")
    for i, file in enumerate(files, 1):
        print(f"{i}) {file}")
    choice = int(input("Enter the number corresponding to your choice: "))
    return files[choice - 1]

# Function to get SMILES files in current directory
def get_smiles_files():
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.smi')]
    return files

# Read SMILES file
smiles_files = get_smiles_files()
dataset_file = prompt_for_file(smiles_files, "dataset")
dataset = read_smiles_file(dataset_file)

# Prompt user for descriptor choice
options = ["Morgan fingerprints", "RDKit fingerprints", "MACCS fingerprints", "RDKit molecular properties", "SiRMS descriptors"]
option_type = "descriptor"
chosen_option, choice_number = prompt_for_option(options, option_type)

# Calculate descriptors
descriptors = []
num_bits = None
if chosen_option in ["Morgan fingerprints", "RDKit fingerprints", "MACCS fingerprints"]:
    for smiles, name in dataset:
        mol = Chem.MolFromSmiles(smiles)
        descriptor = calculate_fingerprints(mol, chosen_option.split()[0])
        if num_bits is None:
            num_bits = descriptor.GetNumBits() if chosen_option == "Morgan fingerprints" else len(descriptor)
        descriptors.append([name] + list(descriptor))
elif chosen_option == "RDKit molecular properties":
    for smiles, name in dataset:
        mol = Chem.MolFromSmiles(smiles)
        descriptor = calculate_molecular_properties(mol)
        qed = calculate_qed(mol)
        descriptors.append([name] + [descriptor[key] for key in sorted(descriptor.keys())] + [qed["QED"]])
else:  # SiRMS descriptors
    sdf_file = 'input.sdf'
    convert_smiles_to_sdf(dataset_file, sdf_file)     
    for smiles, name in dataset:
        descriptor = calculate_sirms_descriptors(sdf_file)
        descriptors.append([name] + [descriptor[key] for key in sorted(descriptor.keys())])

# Create DataFrame
if chosen_option == "SiRMS descriptors":
    descriptor_names = ['Name'] + [desc for desc in sorted(descriptor.keys())]  # Using SiRMS descriptor names
    df = pd.DataFrame(descriptors, columns=descriptor_names)
else:
    if chosen_option in ["Morgan fingerprints", "RDKit fingerprints", "MACCS fingerprints"]:
        descriptor_names = ['Name'] + [f'Bit_{i}' for i in range(num_bits)]
    else:
        descriptor_names = ['Name'] + [desc[0] for desc in Descriptors.descList] + ["QED"] # Using RDKit descriptor names
    df = pd.DataFrame(descriptors, columns=descriptor_names)

# Write DataFrame to CSV
if chosen_option == "RDKit molecular properties":
    output_file = 'output_rdkitproperties.csv'
else:
    output_file = f'output_{chosen_option.lower().replace(" ", "")}.csv'
df.to_csv(output_file, index=False)

print(f"\nDescriptors have been calculated and saved to '{output_file}'.\n")

