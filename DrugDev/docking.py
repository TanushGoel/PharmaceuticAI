from vina import Vina # (AutoDock) Vina forcefield
import subprocess

def dock(center, box_size=[20, 20, 20], exhaust=50):
    v = Vina(sf_name='vina')
    v.set_receptor('temp/prot.pdbqt')
    v.set_ligand_from_file('temp/mol.pdbqt')
    v.compute_vina_maps(center=center, box_size=box_size)
    v.optimize()
    v.dock(exhaustiveness=exhaust, n_poses=20)
    energies = v.energies()
    print(sum(row[0] for row in energies) / len(energies)) # affinity total (kcal/mol)
    # v.write_poses('temp/poses.pdbqt', n_poses=5, overwrite=True)
    
def rmtemp():
    subprocess.run("rm temp/*", shell=True, capture_output=False, text=False)
    
def calc_box(min_coord_obj1, max_coord_obj1, min_coord_obj2, max_coord_obj2, padding=10):
    center_obj1 = [(min + max) / 2 for min, max in zip(min_coord_obj1, max_coord_obj1)]
    center_obj2 = [(min + max) / 2 for min, max in zip(min_coord_obj2, max_coord_obj2)]

    center_x = (center_obj1[0] + center_obj2[0]) / 2
    center_y = (center_obj1[1] + center_obj2[1]) / 2
    center_z = (center_obj1[2] + center_obj2[2]) / 2

    half_length_x = max(abs(center_obj1[0] - center_x) + (max_coord_obj1[0] - min_coord_obj1[0]) / 2,
                       abs(center_obj2[0] - center_x) + (max_coord_obj2[0] - min_coord_obj2[0]) / 2)
    half_length_y = max(abs(center_obj1[1] - center_y) + (max_coord_obj1[1] - min_coord_obj1[1]) / 2,
                       abs(center_obj2[1] - center_y) + (max_coord_obj2[1] - min_coord_obj2[1]) / 2)
    half_length_z = max(abs(center_obj1[2] - center_z) + (max_coord_obj1[2] - min_coord_obj1[2]) / 2,
                       abs(center_obj2[2] - center_z) + (max_coord_obj2[2] - min_coord_obj2[2]) / 2)

    return (center_x, center_y, center_z), (half_length_x + padding, half_length_y + padding, half_length_z + padding)

BOX_ERR_MSG = "Vina runtime error: The ligand is outside the grid box. Increase the size of the grid box or center it accordingly around the ligand."
def autodock(prot, mol):

    try:
        min_coords_mol, max_coords_mol = mol.to_pdbqt()
        min_coords_prot, max_coords_prot = prot.to_pdbqt()
    except:
        print("error in filehandling")
        return None
    
    pad = 5
    for _ in range(5):
        pad += 5
        center, box_size = calc_box(min_coords_mol, max_coords_mol, min_coords_prot, max_coords_prot, pad)
        
        result = subprocess.run(
            ['python', '-c', f'from docking import dock; dock({center}, {box_size})'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        try:
            return float(result.stdout.strip().split('\n')[-1])
        except:
            err = result.stderr.split("Error:")[-1].strip()
            if BOX_ERR_MSG in err:
                print("error in bounding box -->", "Center:", center, "Box Size:", box_size)
            else:
                print("error in docking:", err)
                return None
                
    print("giving up")
 
# Errors:   
# error in docking: PDBQT parsing error: Unknown or inappropriate tag found in flex residue or ligand.