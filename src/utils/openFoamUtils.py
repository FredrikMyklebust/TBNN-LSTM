from scipy.interpolate import interp1d
import re
import sys
import os
import fluidfoam
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil



def ReynoldsStressTensor(uu,uv,uw,vv,vw,ww):
    R = np.empty((len(uu),3,3))
    R[:,0,0] = uu
    R[:,0,1] = uv
    R[:,0,2] = uw
    R[:,1,1] = vv
    R[:,1,2] = vw
    R[:,2,2] = ww
    R[:,1,0] = R[:,0,1]
    R[:,2,0] = R[:,0,2]
    R[:,2,1] = R[:,1,2]
    return R

def write_tensor_to_file(case_path, time_step, tensor, tensor_name, dimensions):
    filename = os.path.join(case_path, time_step, "turbulenceProperties:R")
    
    with open(filename, 'r') as f:
        content = f.read()
        
        header_match = re.search(r'(FoamFile[\s\S]*?)\ndimensions[\s\S]*?;', content)
        if header_match:
            header = header_match.group(1)
            header = re.sub(r'location\s*".*?"', f'location    "{time_step}"', header)
            header = re.sub(r'object\s*.*?;', f'object      {tensor_name};', header)
        else:
            raise ValueError("Header not found in turbulenceProperties:R file")
        
        boundary_field_match = re.search(r'boundaryField\s*\{([\s\S]*)\}', content)
        if boundary_field_match:
            boundary_field = boundary_field_match.group(1)
        else:
            raise ValueError("boundaryField not found in turbulenceProperties:R file")
    
    output_filename = os.path.join(case_path, time_step, tensor_name)
    N = tensor.shape[0]
    
    with open(output_filename, 'w') as f:
        f.write(header)
        f.write(f"\ndimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<symmTensor> \n")
        f.write(f"{N}\n(\n")
        for i in range(N):
            t = tensor[i, :, :]
            f.write(f"({t[0,0]} {t[0,1]} {t[0,2]} {t[1,1]} {t[1,2]} {t[2,2]})\n")
        f.write(")\n;\n")
        f.write(f"boundaryField\n{{\n{boundary_field}\n}}")
        
    #print(f"{tensor_name} file has been written to {output_filename}")

def write_scalar_to_file(case_path, time_step, field, field_name, dimensions):
    filename = os.path.join(case_path, time_step, "k")
    
    with open(filename, 'r') as f:
        content = f.read()
        
        header_match = re.search(r'(FoamFile[\s\S]*?)\ndimensions[\s\S]*?;', content)
        if header_match:
            header = header_match.group(1)
            header = re.sub(r'location\s*".*?"', f'location    "{time_step}"', header)
            header = re.sub(r'object\s*.*?;', f'object      {field_name};', header)
        else:
            raise ValueError("Header not found in k file")
        
        boundary_field_match = re.search(r'boundaryField\s*\{([\s\S]*)\}', content)
        if boundary_field_match:
            boundary_field = boundary_field_match.group(1)
        else:
            raise ValueError("boundaryField not found in k file")
    
    output_filename = os.path.join(case_path, time_step, field_name)
    N = len(field)
    
    with open(output_filename, 'w') as f:
        f.write(header)
        f.write(f"\ndimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<scalar> \n")
        f.write(f"{N}\n(\n")
        for value in field:
            f.write(f"{value}\n")
        f.write(")\n;\n")
        f.write(f"boundaryField\n{{\n{boundary_field}\n}}")
        
    #print(f"{field_name} file has been written to {output_filename}")

def write_vector_to_file(case_path, time_step, field, field_name, dimensions):
    filename = os.path.join(case_path, time_step, "U")
    
    with open(filename, 'r') as f:
        content = f.read()
        
        header_match = re.search(r'(FoamFile[\s\S]*?)\ndimensions[\s\S]*?;', content)
        if header_match:
            header = header_match.group(1)
            header = re.sub(r'location\s*".*?"', f'location    "{time_step}"', header)
            header = re.sub(r'object\s*.*?;', f'object      {field_name};', header)
        else:
            raise ValueError("Header not found in U file")
        
        boundary_field_match = re.search(r'boundaryField\s*\{([\s\S]*)\}', content)
        if boundary_field_match:
            boundary_field = boundary_field_match.group(1)
        else:
            raise ValueError("boundaryField not found in U file")
    
    output_filename = os.path.join(case_path, time_step, field_name)
    N = len(field)
    
    with open(output_filename, 'w') as f:
        f.write(header)
        f.write(f"\ndimensions      {dimensions};\n")
        f.write("internalField   nonuniform List<vector> \n")
        f.write(f"{N}\n(\n")
        for vector in field:
            f.write(f"({vector[0]} {vector[1]} {vector[2]})\n")
        f.write(")\n;\n")
        f.write(f"boundaryField\n{{\n{boundary_field}\n}}")
        
    #print(f"{field_name} file has been written to {output_filename}")



class FluidSimulationData:
    def __init__(self, viscosity, wave_period, stokes_layer_thickness, max_streamwise_velocity):
        """
        Initialize the FluidSimulationData class.
        """
        self.viscosity = viscosity
        self.wave_period = wave_period
        self.stokes_layer_thickness = stokes_layer_thickness
        self.max_streamwise_velocity = max_streamwise_velocity  # Max of the streamwise velocity oscillations
        self.simulation_data = {}  # Holds all simulation data

    def load_dns_data_for_reynolds(self, Reynoldsnumber):
        """
        Load DNS data files for a specific Reynolds number, calculate turbulent kinetic energy and 
        Reynolds stress tensor, and store it in simulation_data.

        Parameters:
        Reynoldsnumber (int): The Reynolds number for which DNS data is to be loaded.
        """
        
        dnsPath = f"/home/fredrik/Documents/research/TemporalFlowNet/data/DNS/vanDerA2018/data_num/DNSRE{str(Reynoldsnumber)}"
        dns_data = {}

        # Read the different components of DNS data
        with nc.Dataset(f"{dnsPath}_u.nc", 'r') as dns_u:

            u2 = dns_u['u2'][:]*self.max_streamwise_velocity ** 2 
            dns_data['u']= dns_u['u'][:]*self.max_streamwise_velocity
            dns_data['yu'] = np.squeeze(dns_u['yu'][:])*self.stokes_layer_thickness
            dns_data['u2'] = u2
        with nc.Dataset(f"{dnsPath}_v.nc", 'r') as dns_v:
            v2 = dns_v['v2'][:]*self.max_streamwise_velocity ** 2
            dns_data['v2'] = v2 
            dns_data['yv'] = np.squeeze(dns_v['yv'][:])*self.stokes_layer_thickness
        with nc.Dataset(f"{dnsPath}_w.nc", 'r') as dns_w:
            w2 = dns_w['w2'][:] *self.max_streamwise_velocity ** 2 
            dns_data['w2'] = w2
            dns_data['yw'] = np.squeeze(dns_w['yw'][:])*self.stokes_layer_thickness
        with nc.Dataset(f"{dnsPath}_uv.nc", 'r') as dns_uv:
            uv = dns_uv['uv'][:] *self.max_streamwise_velocity ** 2 
            dns_data['uv'] = uv

        # Calculate turbulent kinetic energy
        k = 0.5 * (u2 + v2 + w2)
        dns_data['k'] = k

        # Create Reynolds stress tensor (assuming uw and vw components are available)
        # tau = [[u2, uv, uw], [vu, v2, vw], [wu, wv, w2]]
        # Assuming symmetry of the Reynolds stress tensor (vu = uv, wu = uw, wv = vw)
        tau = np.array([[u2, uv, np.zeros_like(u2)],  # uw component is assumed to be zero for simplicity
                        [uv, v2, np.zeros_like(v2)],  # vw component is assumed to be zero for simplicity
                        [np.zeros_like(w2), np.zeros_like(w2), w2]])
        dns_data['tau'] = tau.transpose(2, 3, 0, 1)


        # Store DNS data in simulation_data
        self.simulation_data['DNS'] = dns_data

    def read_field(self, read_func, case_path, time_step, field_name):
        """
        Tries to read a field using the provided read_func. 
        If unsuccessful, returns None.

        Parameters:
        read_func: Function to read the field (readscalar, readvector, readtensor)
        case_path (str): Path to the case directory
        time_step (str): The time step directory
        field_name (str): The name of the field to be read
        """
        try:
            return read_func(case_path, time_step, field_name, verbose=False)
        except FileNotFoundError:
            return None

    def read_all_time_steps(self, case_path, simKey):
        """
        Read all time steps from an OpenFOAM case and store it in simulation_data.
        Handles the absence of certain fields gracefully.

        Parameters:
        case_path (str): The file path to the OpenFOAM case.
        key (str): The key under which to store the data in simulation_data.
        """
        def is_numeric_dir(d):
            try:
                float(d)
                return True
            except ValueError:
                return False

        time_dirs = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d)) and is_numeric_dir(d) and d != "0"]
        time_dirs.sort(key=float)

        # Initialize data storage
        feauters = ['I1_1', 'I1_3', 'I1_4','I1_5', 'I1_7','I1_10', 'I1_13','I1_16','I1_21', 'I1_25','I1_29', 'I1_33','I1_35','I1_43','I1_44']
        data = {'u': [], 'k': [], 'k_DNS': [], 'kDeficit': [], 'PkDelta': [], 'U_DNS': [],
                'Pk': [], 'bijDelta': [], 'I1_1': [], 'I1_3': [],'I1_4': [] , 'I1_5': [], 'I1_7': [],'I1_10': [],
                'I1_13': [],'I1_16': [],'I1_21': [],'I1_25': [],'I1_29': [], 'I1_33': [],'I1_35': [],'I1_43': [],'I1_44': [], 'q2': [], 'q3': [], 'q4': [], 'T1': [],
                'T2': [], 'T3': [], 'T4': [], 'T5': [], 'T6': [], 'T7': [], 'T8': [], 'T9': [], 'T10': []}
        times = []

        for time_step in time_dirs:
            data['u'].append(self.read_field(fluidfoam.readvector, case_path, time_step, 'U')[0,:])
            data['k'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'k'))
            # Repeat for other fields
            data['k_DNS'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'k_DNS'))
            data['kDeficit'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'kDeficit'))
            #data['PkDelta'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'PkDelta'))
            data['U_DNS'].append(self.read_field(fluidfoam.readvector, case_path, time_step, 'U_DNS'))
            #data['Pk'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'Pk'))
            
            for feature in feauters:
                data[feature].append(self.read_field(fluidfoam.readscalar, case_path, time_step, feature))
            data['q2'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'q2'))
            data['q3'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'q3'))
            data['q4'].append(self.read_field(fluidfoam.readscalar, case_path, time_step, 'q4'))
            #reshape the tensor to 3x3
            bijDelta = self.read_field(fluidfoam.readsymmtensor, case_path, time_step, 'bijDelta')
            if bijDelta is not None:
                N = bijDelta.shape[1]
                tensor_3D = np.zeros((N, 3, 3))
                tensor_3D[:, 0, 0] = bijDelta[0, :]
                tensor_3D[:, 0, 1] = tensor_3D[:, 1, 0] = bijDelta[1, :]
                tensor_3D[:, 0, 2] = tensor_3D[:, 2, 0] = bijDelta[2, :]
                tensor_3D[:, 1, 1] = bijDelta[3, :]
                tensor_3D[:, 1, 2] = tensor_3D[:, 2, 1] = bijDelta[4, :]
                tensor_3D[:, 2, 2] = bijDelta[5, :]
                data['bijDelta'].append(tensor_3D)

            T1 = self.read_field(fluidfoam.readsymmtensor, case_path, time_step, 'T1')
            # check if T1 is not None
            if T1 is not None:
                N = T1.shape[1]
                tensor_3D = np.zeros((N, 3, 3))
                tensor_3D[:, 0, 0] = T1[0, :]
                tensor_3D[:, 0, 1] = tensor_3D[:, 1, 0] = T1[1, :]
                tensor_3D[:, 0, 2] = tensor_3D[:, 2, 0] = T1[2, :]
                tensor_3D[:, 1, 1] = T1[3, :]
                tensor_3D[:, 1, 2] = tensor_3D[:, 2, 1] = T1[4, :]
                tensor_3D[:, 2, 2] = T1[5, :]
                data['T1'].append(tensor_3D)
                for i in range(2,11):
                    data[f'T{i}'].append(self.read_field(fluidfoam.readtensor, case_path, time_step, f'T{i}').transpose(1,0).reshape(-1,3,3))
            
            times.append(float(time_step))
        # Convert lists to numpy arrays and handle None values
        for key, value in data.items():
            data[key] = np.array([np.nan if elem is None else elem for elem in value])
        x, y, z = fluidfoam.readmesh(case_path, verbose=False)
        try:
            data['U_DNS'] = data['U_DNS'][:,0,:]
        except:
            pass
        # Store OpenFOAM data in simulation_data
        self.simulation_data[simKey] = {'times': np.array(times), 'x': x, 'y': y, 'z': z, **data}
    """
    def read_all_time_steps(self, case_path, key):
        Read all time steps from an OpenFOAM case and store it in simulation_data.

        Parameters:
        case_path (str): The file path to the OpenFOAM case.
        key (str): The key under which to store the data in simulation_data.
        
        def is_numeric_dir(d):
            try:
                float(d)
                return True
            except ValueError:
                return False

        time_dirs = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d)) and is_numeric_dir(d) and d != "0"]
        time_dirs.sort(key=float)

        u = []
        k= []
        times = []

        for time_step in time_dirs:
            U = fluidfoam.readvector(case_path, time_step, 'U',verbose=False)
            k_read = fluidfoam.readscalar(case_path, time_step, 'k',verbose=False)
            u.append(U[0, :])  # Assuming U[0, :] is the x component
            k.append(k_read)
            times.append(float(time_step))
        x, y, z = fluidfoam.readmesh(case_path, verbose=False)

        u = np.array(u)
        times = np.array(times)
        k= np.array(k)

        # Store OpenFOAM data in simulation_data
        self.simulation_data[key] = {'times': times, 'u': u, 'k': k, 'y': y}
    """
    def interpolate_dns_to_openfoam(self, dns_key='DNS', openfoam_key='OpenFOAM', new_key='Interpolated'):
        """
        Interpolate DNS data to match OpenFOAM mesh and store it in simulation_data.

        Parameters:
        dns_key (str): The key for DNS data in simulation_data.
        openfoam_key (str): The key for OpenFOAM data in simulation_data.
        new_key (str): The key under which to store the interpolated data.
        """
        if dns_key not in self.simulation_data or openfoam_key not in self.simulation_data:
            print("Required DNS or OpenFOAM data not found.")
            return

        dns_data = self.simulation_data[dns_key]
        openfoam_data = self.simulation_data[openfoam_key]
        y_openfoam = openfoam_data['y']

        interpolated_data = {}

        # Interpolate each component
        u_ = interp1d(dns_data['yu'], dns_data['u'], kind='linear', fill_value='extrapolate')(y_openfoam)
        interpolated_data['u'] = u_
        U = np.array([u_, np.zeros_like(u_), np.zeros_like(u_)])
        interpolated_data['U'] = U.transpose(1, 2, 0)
        interpolated_data['u2'] = interp1d(dns_data['yu'], dns_data['u2'], kind='linear', fill_value='extrapolate')(y_openfoam)
        interpolated_data['uv'] = interp1d(dns_data['yu'], dns_data['uv'], kind='linear', fill_value='extrapolate')(y_openfoam)
        interpolated_data['uw'] = np.zeros_like(interpolated_data['uv'])
        interpolated_data['v2'] = interp1d(dns_data['yv'], dns_data['v2'], kind='linear', fill_value='extrapolate')(y_openfoam)
        interpolated_data['vw'] = np.zeros_like(interpolated_data['uv'])
        interpolated_data['w2'] = interp1d(dns_data['yw'], dns_data['w2'], kind='linear', fill_value='extrapolate')(y_openfoam)

        interpolated_data['y'] = y_openfoam
        tau = np.array([[interpolated_data['u2'], interpolated_data['uv'], np.zeros_like(interpolated_data['u2'])],  # uw component is assumed to be zero for simplicity
                        [interpolated_data['uv'], interpolated_data['v2'], np.zeros_like(interpolated_data['v2'])],  # vw component is assumed to be zero for simplicity
                        [np.zeros_like(interpolated_data['w2']), np.zeros_like(interpolated_data['w2']), interpolated_data['w2']]])
        interpolated_data['tau'] = tau.transpose(2, 3, 0, 1)
        interpolated_data['k'] = 0.5 * (interpolated_data['u2'] + interpolated_data['v2'] + interpolated_data['w2'])
        interpolated_data['bij'] = interpolated_data['tau']/(2*interpolated_data['k'][:,:,None,None]) - 1/3*np.identity(3)
        interpolated_data['aij'] = interpolated_data['tau'] - 2/3*np.trace(interpolated_data['tau'])*np.identity(3)
        # Store interpolated data in simulation_data
        self.simulation_data[new_key] = interpolated_data
        
    def plot_data_at_timestep(self, timestep, dns_key='DNS', openfoam_keys=['OpenFOAM'], primary_component='u', secondary_component=None):
        """
        Plot DNS and multiple OpenFOAM datasets for a specific component over the y-axis at a given timestep.
        Optionally plots a second component from OpenFOAM data if available. Designed for scientific publications.

        Parameters:
        timestep (float): The specific timestep to plot data for.
        dns_key (str): The key for DNS data in simulation_data.
        openfoam_keys (list of str): The list of keys for OpenFOAM data in simulation_data.
        primary_component (str): The primary data component to plot (e.g., 'u', 'k').
        secondary_component (str, optional): The secondary data component from OpenFOAM data to plot if available.
        """
        # Set style for scientific plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        # Check if DNS data is available
        if dns_key not in self.simulation_data:
            print(f"DNS data not available for key {dns_key}.")
            return

        dns_data = self.simulation_data[dns_key]

        # Check if the DNS component and y-values are available
        if primary_component not in dns_data or 'yu' not in dns_data:
            print(f"Component '{primary_component}' or 'yu' not available in DNS data.")
            return

        dns_plot_data = dns_data[primary_component][timestep]
        dns_y_values = dns_data['yu']

        # Plot DNS data
        plt.plot(dns_plot_data, dns_y_values, label=f'DNS {primary_component} at timestep {timestep}', marker='o', linestyle='-')

        # Plot settings for OpenFOAM data
        of_markers = ['s', '^', 'd', 'x', '*']  # Different markers for different OpenFOAM datasets
        of_linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # Different line styles

        # Iterate over OpenFOAM keys
        for idx, openfoam_key in enumerate(openfoam_keys):
            # Check if OpenFOAM data is available for the current key
            if openfoam_key not in self.simulation_data:
                print(f"OpenFOAM data not available for key {openfoam_key}.")
                continue

            openfoam_data = self.simulation_data[openfoam_key]

            # Check if the OpenFOAM primary component and y-values are available
            if primary_component not in openfoam_data or 'y' not in openfoam_data:
                print(f"Component '{primary_component}' or 'y' not available in OpenFOAM data for key {openfoam_key}.")
                continue

            openfoam_primary_plot_data = openfoam_data[primary_component][timestep]
            openfoam_y_values = openfoam_data['y']

            # Plot OpenFOAM primary component data
            plt.plot(openfoam_primary_plot_data, openfoam_y_values, label=f'OpenFOAM ({openfoam_key}) {primary_component} at timestep {timestep}', marker=of_markers[idx % len(of_markers)], linestyle=of_linestyles[idx % len(of_linestyles)])

            # Plot secondary component for OpenFOAM if available
            if secondary_component and secondary_component in openfoam_data:
                openfoam_secondary_plot_data = openfoam_data[secondary_component][timestep]
                plt.plot(openfoam_secondary_plot_data, openfoam_y_values, label=f'OpenFOAM ({openfoam_key}) {secondary_component} at timestep {timestep}', linestyle=':', marker=of_markers[idx % len(of_markers)])

        # Finalizing plot
        plt.ylabel('Y Dimension')
        plt.xlabel(f'{primary_component} Value')
        plt.title(f'Comparison of {primary_component} Component at Timestep {timestep}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def write_simulation_data(case_path, u, k, tau):
    """
    Writes simulation data (u, k, tau) to each time step folder.

    :param case_path: Path to the OpenFOAM case directory
    :param u: Velocity data array (700, n, 3)
    :param k: Turbulent kinetic energy data array (700, n)
    :param tau: Stress tensor data array (700, n, 3, 3)
    """
    # List all time step directories
    time_dirs = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d)) and is_numeric_dir(d) and d != "0"]
    time_dirs.sort(key=float)

    # Loop over each time step directory
    for i, time_dir in enumerate(time_dirs):

        # Check if the time step index is within the bounds of the data arrays
        if i < len(u) and i < len(k) and i < len(tau):
            time_dir_path = os.path.join(case_path, time_dir)
            
            # Write data to the time step directory
            write_vector_to_file(case_path, time_dir, u[i], "Udns", "[0 1 -1 0 0 0 0]")
            write_scalar_to_file(case_path, time_dir, k[i], "kdns", "[0 2 -2 0 0 0 0]")
            write_tensor_to_file(case_path, time_dir, tau[i], "Rdns", "[0 2 -2 0 0 0 0]")
            rename_files(time_dir_path)


def rename_files(time_dir_path):
    """
    Renames the existing files in the directory.
    
    :param time_dir_path: Path to the time step directory
    """
    file_mapping = {"k": "krans", "U": "Urans", "turbulenceProperties:R": "Rrans"}
    for old_name, new_name in file_mapping.items():
        old_path = os.path.join(time_dir_path, old_name)
        new_path = os.path.join(time_dir_path, new_name)
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)

def is_numeric_dir(d):
    """
    Check if the directory name is a number (time step)
    """
    try:
        float(d)
        return True
    except ValueError:
        return False
    
def fullTensor(symmtensor):
    N = symmtensor.shape[1]
    tensor_3D = np.zeros((N, 3, 3))
    tensor_3D[:, 0, 0] = symmtensor[0, :]
    tensor_3D[:, 0, 1] = tensor_3D[:, 1, 0] = symmtensor[1, :]
    tensor_3D[:, 0, 2] = tensor_3D[:, 2, 0] = symmtensor[2, :]
    tensor_3D[:, 1, 1] = symmtensor[3, :]
    tensor_3D[:, 1, 2] = tensor_3D[:, 2, 1] = symmtensor[4, :]
    tensor_3D[:, 2, 2] = symmtensor[5, :]
    return tensor_3D

import re
import os

def write_reynolds_stress_file(input_file_path, time_name, reynolds_stress_tensor, output_file_path):
    """
    Reads the 'turbulenceProperties:R' file from the input path, and writes the Reynolds stress tensor to the output file path.

    :param input_file_path: Path to the 'turbulenceProperties:R' file.
    :param time_name: Name of the time step for the output directory.
    :param reynolds_stress_tensor: The Reynolds stress tensor data.
    :param output_file_path: Path to write the output file.
    """
    input_file = input_file_path + "/0old/turbulenceProperties:R"
    # Read the input 'turbulenceProperties:R' file
    with open(input_file, 'r') as f:
        content = f.read()
        
        # Find and modify the header
        header_match = re.search(r'(FoamFile[\s\S]*?)\ndimensions[\s\S]*?;', content)
        if header_match:
            header = header_match.group(1)
            header = re.sub(r'location\s*".*?"', f'location    "{time_name}"', header)
            header = re.sub(r'object\s*.*?;', 'object      R;', header)
        else:
            raise ValueError("Header not found in 'turbulenceProperties:R' file")
        
        # Find the boundary field
        boundary_field_match = re.search(r'boundaryField\s*\{([\s\S]*)\}', content)
        if boundary_field_match:
            boundary_field = boundary_field_match.group(1)
        else:
            raise ValueError("boundaryField not found in 'turbulenceProperties:R' file")
    
    # Define the output filename
    output_filename = os.path.join(output_file_path,time_name,"bijDelta")
    N = reynolds_stress_tensor.shape[0]
    
    # Write the Reynolds stress tensor data to the output file
    with open(output_filename, 'w') as f:
        f.write(header)
        f.write("\ndimensions      [0 0 0 0 0 0 0];\n")
        f.write("internalField   nonuniform List<symmTensor> \n")
        f.write(f"{N}\n(\n")
        for i in range(N):
            t = reynolds_stress_tensor[i, :, :]
            f.write(f"({t[0,0]} {t[0,1]} {t[0,2]} {t[1,1]} {t[1,2]} {t[2,2]})\n")
        f.write(")\n;\n")
        f.write(f"boundaryField\n{{\n{boundary_field}\n}}")
        
    print(f"Reynolds stress file has been written to {output_filename}")

def list_and_write_reynolds_stress(input_file_path, output_file_path, reynolds_stress_tensor):
    """
    Lists directories, adjusts time steps, and writes Reynolds stress tensor to output path.

    :param input_file_path: Path where the input directories are located.
    :param output_file_path: Path where the Reynolds stress tensor files should be written.
    :param reynolds_stress_tensor: The Reynolds stress tensor data.
    """

    def is_numeric_dir(d):
        try:
            float(d)
            return True
        except ValueError:
            return False

    # List and sort directories
    time_dirs = [d for d in os.listdir(input_file_path) if os.path.isdir(os.path.join(input_file_path, d)) and is_numeric_dir(d) and d != "0"]
    time_dirs.sort(key=lambda x: float(x) - 1400)

    # Process each time step
    for i, time_dir in enumerate(time_dirs):
        adjusted_time_step = float(time_dir) - 1400
        adjusted_time_name = "{:.2f}".format(adjusted_time_step)
        adjusted_time_name = "{:.2f}".format(adjusted_time_step).rstrip('0').rstrip('.')
        adjusted_dir_path = os.path.join(output_file_path, adjusted_time_name)

        # Create directory if it doesn't exist
        if not os.path.exists(adjusted_dir_path):
            os.makedirs(adjusted_dir_path)

        # Write the Reynolds stress tensor file
        write_reynolds_stress_file(input_file_path, adjusted_time_name, reynolds_stress_tensor[i], output_file_path)
    #lets also write the last one to 0 
    #also have to create the directory
    if not os.path.exists(os.path.join(output_file_path, "0")):
        os.makedirs(os.path.join(output_file_path, "0"))
    write_reynolds_stress_file(input_file_path, "0", reynolds_stress_tensor[-1], output_file_path)

def write_kDeficit_file(input_file_path, time_name, reynolds_stress_tensor, output_file_path):
    """
    Reads the 'turbulenceProperties:R' file from the input path, and writes the Reynolds stress tensor to the output file path.

    :param input_file_path: Path to the 'turbulenceProperties:R' file.
    :param time_name: Name of the time step for the output directory.
    :param reynolds_stress_tensor: The Reynolds stress tensor data.
    :param output_file_path: Path to write the output file.
    """
    input_file = input_file_path + "/0old/k"
    # Read the input 'k' file
    with open(input_file, 'r') as f:
        content = f.read()
        
        # Find and modify the header
        header_match = re.search(r'(FoamFile[\s\S]*?)\ndimensions[\s\S]*?;', content)
        if header_match:
            header = header_match.group(1)
            header = re.sub(r'location\s*".*?"', f'location    "{time_name}"', header)
            header = re.sub(r'object\s*.*?;', 'object      k;', header)
        else:
            raise ValueError("Header not found in 'k' file")
        
        # Find the boundary field
        boundary_field_match = re.search(r'boundaryField\s*\{([\s\S]*)\}', content)
        if boundary_field_match:
            boundary_field = boundary_field_match.group(1)
        else:
            raise ValueError("boundaryField not found in 'k' file")
    
    # Define the output filename
    output_filename = os.path.join(output_file_path,time_name,"kDeficit")
    N = reynolds_stress_tensor.shape[0]
    
    # Write the Reynolds stress tensor data to the output file
    with open(output_filename, 'w') as f:
        f.write(header)
        f.write("\ndimensions      [0 2 -3 0 0 0 0];\n")
        f.write("internalField   nonuniform List<scalar> \n")
        f.write(f"{N}\n(\n")
        for i in range(N):
            t = reynolds_stress_tensor[i][0]
            f.write(f"{t}\n")
        f.write(")\n;\n")
        f.write(f"boundaryField\n{{\n{boundary_field}\n}}")
        
    print(f"Reynolds stress file has been written to {output_filename}")
def list_and_write_kDeficit(input_file_path, output_file_path, reynolds_stress_tensor):
    """
    Lists directories, adjusts time steps, and writes Reynolds stress tensor to output path.

    :param input_file_path: Path where the input directories are located.
    :param output_file_path: Path where the Reynolds stress tensor files should be written.
    :param reynolds_stress_tensor: The Reynolds stress tensor data.
    """

    def is_numeric_dir(d):
        try:
            float(d)
            return True
        except ValueError:
            return False

    # List and sort directories
    time_dirs = [d for d in os.listdir(input_file_path) if os.path.isdir(os.path.join(input_file_path, d)) and is_numeric_dir(d) and d != "0"]
    time_dirs.sort(key=lambda x: float(x) - 1400)

    # Process each time step
    for i, time_dir in enumerate(time_dirs):
        adjusted_time_step = float(time_dir) - 1400
        adjusted_time_name = "{:.2f}".format(adjusted_time_step)
        adjusted_time_name = "{:.2f}".format(adjusted_time_step).rstrip('0').rstrip('.')
        adjusted_dir_path = os.path.join(output_file_path, adjusted_time_name)

        # Create directory if it doesn't exist
        if not os.path.exists(adjusted_dir_path):
            os.makedirs(adjusted_dir_path)

        # Write the Reynolds stress tensor file
        write_kDeficit_file(input_file_path, adjusted_time_name, reynolds_stress_tensor[i], output_file_path)
    #lets also write the last one to 0 
    #also have to create the directory
    if not os.path.exists(os.path.join(output_file_path, "0")):
        os.makedirs(os.path.join(output_file_path, "0"))
    write_kDeficit_file(input_file_path, "0", reynolds_stress_tensor[-1], output_file_path)

def read_wall_shear_stress_with_regex(filename):
    """
    Reads an OpenFOAM file to extract the wall shear stress value for the 'bottom' boundary using regex.

    Parameters:
    filename (str): The path to the file to be read.

    Returns:
    float: The wall shear stress value at the 'bottom' boundary, or None if not found.
    """
    # Regex pattern to find the wall shear stress value for the 'bottom' boundary
    pattern = r'bottom\s*{[^}]*value\s*uniform\s*\(([^)]*)\);'
    
    # Read the entire file content
    with open(filename, 'r') as file:
        file_content = file.read()
        
    # Search for the pattern in the file content
    match = re.search(pattern, file_content, re.DOTALL)
    
    if match:
        # Extract the matched group, which contains the shear stress values
        values_str = match.group(1)
        # Split the values and convert the first one to float
        values = values_str.split()
        wall_shear_stress = float(values[0])
        return wall_shear_stress
    
    # Return None if the pattern is not found
    return None