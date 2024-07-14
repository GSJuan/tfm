import os
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


def visualize_molecules(input_smiles, generated_smiles, output_folder='images'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate molecule objects
    input_molecule = Chem.MolFromSmiles(input_smiles)
    generated_molecule = Chem.MolFromSmiles(generated_smiles)

    # Create images
    input_img = Draw.MolToImage(input_molecule)
    generated_img = Draw.MolToImage(generated_molecule)

    # Display the images inline using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display input molecule
    axes[0].imshow(input_img)
    axes[0].set_title("Sampled Molecule")
    axes[0].axis('off')

    # Display generated molecule
    axes[1].imshow(generated_img)
    axes[1].set_title("Generated Molecule")
    axes[1].axis('off')

    # Save the combined image
    combined_image_path = os.path.join(output_folder, 'molecules.png')
    plt.tight_layout()
    plt.savefig(combined_image_path)
    plt.show()

def visualize_molecule(smiles, output_folder="images"):
    # Generate the molecule from SMILES string
    
    molecule = Chem.MolFromSmiles(smiles)

    # Draw the molecule
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    img = Draw.MolToImage(molecule)
    ax.imshow(img)
    ax.axis('off')
    
    combined_image_path = os.path.join(output_folder, 'molecules.png')
    plt.savefig(combined_image_path)
    plt.show()

    
if __name__ == "__main__":
    """
    input_smiles = 'OC1C2NCC3C4C=CC2C1C3O4'
    generated_smiles = 'C=C3CCCCC3N'
    visualize_molecules(input_smiles, generated_smiles)
    """
    smiles = 'CC(=O)O'    
    visualize_molecule(smiles)