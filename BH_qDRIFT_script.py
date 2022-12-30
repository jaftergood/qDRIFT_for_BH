from qiskit.opflow import PauliSumOp # ListOp
from qiskit.quantum_info import SparsePauliOp

from scipy.sparse.linalg import expm_multiply, expm
import numpy as np

import h5py
import argparse

from utils import qDRIFT, propQ, find_gs_vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-t", "--t", type=float, default=1628., help="Final time in attoseconds")
    parser.add_argument("-d", "--d", type=float, default=10., help="Discretization step size")
    parser.add_argument("-m", "--m", type=int, default=1000, help="Number of qDRIFT steps to take")
    parser.add_argument("-a", "--a", type=float, default=0.003, help="Amplitude, the driving amplitude in a.u.")
    parser.add_argument("-o", "--o", type=float, default=0.093368, help="Omega, the driving frequency in a.u.")
    parser.add_argument("-s", "--s", type=str, default='1m', help="Hamiltonian type (there are 4 types: 1p, 1m, 2p, 2m)")
    args = parser.parse_args()
    t = args.t
    dt = args.d
    m = args.m
    a = args.a
    o = args.o
    dif = args.s

    # Load in Hamiltonian and z-dipole data:

    qu_hamiltonian = SparsePauliOp(['IIIIII'], coeffs=[0])
    with open('BH_ham.txt', 'r') as file:
        for line in file:
            A, B = line.split('*')
            qu_hamiltonian += SparsePauliOp([B[0:6]], coeffs=[float(A)])
        qu_hamiltonian = PauliSumOp(qu_hamiltonian).reduce()
        
    qu_dip = SparsePauliOp(['IIIIII'], coeffs=[0])
    with open('BH_dip.txt', 'r') as file:
        for line in file:
            A, B = line.split('*')
            qu_dip += SparsePauliOp([B[0:6]], coeffs=[float(A)])
        qu_dip = PauliSumOp(qu_dip).reduce()

    #This hamiltonian is in the Molecular Orbital basis
    hamiltonian = qu_hamiltonian.to_matrix()

    #Diagonalizes the Hamiltonian
    eigvals, eigvecs = np.linalg.eig(hamiltonian)

    #find the eigenvector for the ground state energy
    gs_vec = find_gs_vec(eigvals, eigvecs)

    conv = 24.188843265857

    tf = t/conv; dtf = dt/conv

    propQ(qu_hamiltonian, dif, qu_dip, o, a, gs_vec, tf, dtf, m)
