# Copyright 2025 Sear Gamemode
import mne
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Delaunay


def align_head(ch_dict, nasion, lpa, rpa, hsp, standard='waveguard64', mode='Cz', threshold=0.1):
    def calculate_rotation_matrix(A, B):
        H = A @ B.T
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        return R
    
    def interpolate_electrode_proj(pos, neighbor, hsp):
        tree = KDTree(hsp)
        _, idx = tree.query(neighbor, k=1)
        return hsp[idx]
    
    def interpolate_electrode_tri(ch_dict, standard_pos, hsp):
        tri = Delaunay(hsp)
        simplex = tri.find_simplex(standard_pos)
        if simplex == -1:
            print(f"Electrode {standard_pos} is outside the triangulation")
        interpolated_pos = np.mean(hsp[simplex >= 0], axis=0)
        return interpolated_pos

    def coord_update(ch_dict, nasion, lpa, rpa, hsp, vector):
        ch_dict = {ch: pos+vector for ch, pos in ch_dict.items()} 
        nasion += vector
        lpa += vector
        rpa += vector
        hsp += vector
        return ch_dict, nasion, lpa, rpa, hsp
    def compute_stats(ch_dict):
        coords = np.array(list(ch_dict.values()))
        mean = np.mean(coords, axis=0)
        std = np.std(coords, axis=0)
        return mean, std
    standard_montage = create_custom_montage(standard)
    standard_pos = standard_montage.get_positions()['ch_pos']
    standard_points = np.array([standard_pos['Cz'], standard_pos['M1'], standard_pos['M2'], standard_pos['Fpz']])
    points = np.array([ch_dict['Cz'], ch_dict['M1'], ch_dict['M2'], ch_dict['Fpz']])
    R = calculate_rotation_matrix(points.T, standard_points.T)

    ch_dict = {ch: R.dot(pos) for ch, pos in ch_dict.items()}
    translation_vector = np.mean(list(standard_pos.values())) - np.mean(list(ch_dict.values()))
    nasion = R.dot(nasion)
    lpa = R.dot(lpa)
    rpa = R.dot(rpa)
    hsp = np.dot(hsp, R.T)
    ch_dict, nasion, lpa, rpa, hsp, = coord_update(ch_dict, nasion, lpa, rpa, hsp, translation_vector)

    ind_mean, ind_std = compute_stats(ch_dict)
    st_mean, st_std = compute_stats(standard_pos)

    ch_norm = {ch: ((pos-ind_mean)/ind_std)*st_std + st_mean for ch, pos in ch_dict.items()}
    hsp_norm = np.array([((coord-ind_mean)/ind_std)*st_std + st_mean for coord in hsp])

    for ch in standard_pos.keys():
        ch_error = np.linalg.norm(ch_norm[ch] - standard_pos[ch])
        #print(ch, ch_error)
        if  ch_error > threshold:
            print(f'{ch} interpolated with error={ch_error}')
            #neighbors_individual = np.array([ch_dict[n_ch] for n_ch in ch_dict if n_ch != ch])
            ch_coord = interpolate_electrode_proj(ch_norm[ch], standard_pos[ch], hsp_norm)
            #ch_coord = interpolate_electrode_tri(ch_norm, standard_pos[ch], hsp_norm)
            ch_error = np.linalg.norm(ch_coord- standard_pos[ch])
            print(f'New error={ch_error}')
            ch_dict[ch] = ((ch_coord-st_mean)/st_std)*ind_std + ind_mean
    
    if mode == 'Cz':
        translation_vector = standard_pos['Cz'] - ch_dict['Cz']
        ch_dict, nasion, lpa, rpa, hsp, = coord_update(ch_dict, nasion, lpa, rpa, hsp, translation_vector)
    return ch_dict, nasion, lpa, rpa, hsp

def read_elc(elc_file):
    lines = []
    with open(elc_file, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())

    num_ch = int(lines[0].strip().split('=')[1])
    ch_dict = {}
    for line in lines[3:num_ch+3]:
        ch, coord = line.split(':')
        ch = ch.replace('\t', '')
        coord = list(map(float, coord.split('\t')[1:]))
        coord = [-coord[1]/1000, coord[0]/1000, coord[2]/1000]
        ch_dict[ch]=coord

    num_hsp = int(lines[num_ch+5].strip().split('=')[1])
    hsp = []
    for line in lines[num_ch+8:num_ch+8+num_hsp]:
        coord = list(map(float, line.split('\t')))
        coord = [-coord[1]/1000, coord[0]/1000, coord[2]/1000]
        hsp.append(coord)

    nasion = ch_dict['Nasion']
    lpa = ch_dict['LeftEar']
    rpa = ch_dict['RightEar']
    del ch_dict['EOG']
    del ch_dict['Nasion']
    del ch_dict['LeftEar']
    del ch_dict['RightEar']
    return ch_dict, nasion, lpa, rpa, hsp

def create_custom_montage(montage):
    if montage == 'waveguard64':
        ch_pos = {
        'Fp1': np.array([94.9, 30.7, 14.0]),
        'Fpz': np.array([98.3, -0.2, 18.6]),
        'Fp2': np.array([96.4, -31.5, 14.1]),
        'F7' : np.array([52.0, 74.2, 3.1]),
        'F3' : np.array([52.0, 52.9, 56.7]),
        'Fz' : np.array([52.4, -0.4, 79.2]),
        'F4' : np.array([53.1, -54.9, 55.4]),
        'F8' : np.array([53.6, -76.0, 2.9]),
        'FC5': np.array([18.5, 80.9, 33.5]),
        'FC1': np.array([15.1, 34.7, 87.3]),
        'FC2': np.array([15.8, -35.8, 85.8]),
        'FC6': np.array([19.9, -84.0, 33.3]),
        'M1' : np.array([-37.1, 89.8, -68.3]),
        'T7' : np.array([-11.7, 87.4, -6.2]),
        'C3' : np.array([-22.8, 67.2, 64.9]),
        'Cz' : np.array([-27.5, -0.5, 96.9]),
        'C4' : np.array([-21.9, -69.1, 64.0]),
        'T8' : np.array([-10.6, -88.2, -6.3]),
        'M2' : np.array([-36.9, -88.8, -69.2]),
        'CP5': np.array([-53.1, 82.8, 26.0]),
        'CP1': np.array([-66.8, 37.1, 83.7]),
        'CP2': np.array([-66.3, -39.9, 83.6]),
        'CP6': np.array([-52.4, -86.2, 26.7]),
        'P7' : np.array([-74.9, 75.6, -12.0]),
        'P3' : np.array([-92.6, 55.1, 42.5]),
        'Pz' : np.array([-100.5, -0.4, 66.9]),
        'P4' : np.array([-92.7, -57.3, 42.6]),
        'P8' : np.array([-74.9, -76.2, -12.2]),
        'POz': np.array([-115.9, -0.3, 30.7]),
        'O1' : np.array([-118.8, 30.8, -11.0]),
        'Oz' : np.array([-122.6, -0.2, -7.3]),
        'O2' : np.array([-118.8, -31.2, -11.1]),
        'AF7': np.array([79.3, 57.4, 8.6]),
        'AF3': np.array([82.4, 35.5, 41.0]),
        'AF4': np.array([83.1, -37.7, 41.4]),
        'AF8': np.array([80.3, -57.9, 8.5]),
        'F5' : np.array([52.0, 67.4, 31.3]),
        'F1' : np.array([52.1, 28.5, 74.1]),
        'F2' : np.array([52.7, -31.0, 73.2]),
        'F6' : np.array([53.4, -70.0, 30.8]),
        'FC3': np.array([16.5, 63.1, 64.4]),
        'FCz': np.array([14.8, -0.5, 95.5]),
        'FC4': np.array([17.6, -64.7, 63.4]),
        'C5' : np.array([-17.6, 84.9, 31.6]),
        'C1' : np.array([-26.7, 38.0, 90.8]),
        'C2' : np.array([-26.0, -39.8, 89.2]),
        'C6' : np.array([-16.6, -86.3, 31.4]),
        'CP3': np.array([-61.0, 65.7, 58.9]),
        'CPz': np.array([-68.2, -0.5, 90.2]),
        'CP4': np.array([-60.1, -68.0, 58.5]),
        'P5' : np.array([-84.3, 70.5, 17.0]),
        'P1' : np.array([-98.3, 29.6, 60.2]),
        'P2' : np.array([-98.6, -32.1, 60.8]),
        'P6' : np.array([-84.3, -72.2, 16.9]),
        'PO5': np.array([-107.2, 50.3, 4.9]),
        'PO3': np.array([-112.3, 39.6, 20.4]),
        'PO4': np.array([-112.5, -38.3, 19.4]),
        'PO6': np.array([-107.5, -49.5, 4.1]),
        'FT7': np.array([20.9, 83.6, -1.8]),
        'FT8': np.array([22.4, -85.0, -2.0]),
        'TP7': np.array([-44.5, 87.9, -10.1]),
        'TP8': np.array([-43.5, -87.0, -10.0]),
        'PO7': np.array([-101.5, 57.2, -12.6]),
        'PO8': np.array([-102.1, -58.2, -12.9]),
        }
    for key in ch_pos.keys():
        ch_pos[key][0], ch_pos[key][1] = -ch_pos[key][1], ch_pos[key][0]
        ch_pos[key] /= 1000
    Nasion= [ 5.27205792e-18,  8.60992398e-02, -4.01487349e-02]
    LPA= [-0.08609924, -0., -0.04014873]
    RPA= [ 0.08609924,  0., -0.04014873]
    dig_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=Nasion, lpa=LPA, rpa=RPA, coord_frame='head')
    return dig_montage
