from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import h5py


def qDRIFT(
           H: PauliSumOp,
           dt: float,
           n: int,
          ):
    '''
    Time dependent qDRIFT component.
    
    Breaks up a Hamiltonian into components amenable to qDRIFT and then makes
    operators for qDRIFT on an interval of size dt broken into n steps.
    
    INPUTS:
    
    H  :: Hamiltonian in Qiskit's PauliSumOp class.
    dt :: the time interval on which to perform qDRIFT as a float.
    n  :: an integer number of break-ups to break the interval into.
    
    RETURNS:
    
    A list of operators in PauliSumOp form. Use the to_matrix() method
    to make the operators into usable matrices.
    '''
    # Split Hamiltonian into coefficients and operators
    vals = [str(n.to_pauli_op()).split(' * ') for n in H]
    # Massages the Hamiltonian data into the required form
    coeffs = [np.abs(np.float64(x)) for x, _ in vals]
    # The sum of the (now entirely positive) coefficients
    lam = sum(coeffs)
    # With lam and n we can now define the qDRIFT 'time step'
    tau = lam*dt/n
    # List of operators in PauliSumOp form
    str_ops = [PauliSumOp(SparsePauliOp([y], coeffs=[tau])) if 
              np.float64(x) > 0 else PauliSumOp(SparsePauliOp([y], coeffs=[-tau])) 
              for x, y in vals]
    # Produces the probabilities for this round of qDRIFT
    probs = [cof/lam for cof in coeffs]
    # Produce list of integers 0 to len(H) from probabilities listed in probs
    rand_list = list(np.random.choice([i for i in range(len(H))], int(n), p=probs))
    op_list = [str_ops[m].exp_i() for m in rand_list]
    return op_list


def propQ(
          ham: PauliSumOp,
          dif: str,
          dip_z: PauliSumOp, 
          omeg: float,
          amp: float,
          gs_vec: np.ndarray,
          t_max: float,
          dt: float,
          n: float,
         ):

    '''
    Performs the time evolution using qDRIFT.

    INPUTS:

    ham    :: The base Hamiltonian operator in PauliSumOp form. (Couplings should be in a.u.)
    dif    :: String that selects correct Hamiltonian. Must be 1p, 1m, 2p, or 2m.
    dip_z  :: The (z) dipole operator in PauliSumOp form. (Couplings should be in a.u.)
    omeg   :: The frequency of the driving electric field. (Use a.u.)
    amp    :: The amplitude of the driving electric field. (Use a.u.)
    gs_vec :: The ground-state vector as initial state in np.array form.
    t_max  :: The total time over which to propagate in float form. (Give times in attosecons [converts to a.u. automatically])
    dt     :: The time step (dt < t_max). (Give times in attoseconds [converts to a.u. automatically])
    n      :: The number of qDRIFT steps to take within a time-step.

    RETURNS:

    Creates a .hdf5 file with the wavefunctions at each time in one database and the associated times in another. 

    Function return is None value.

    '''

    # Conversion from attoseconds to a.u.
    conv = 24.188843265857
    # Form the z-dipole operator matrix.
    dip_op = dip_z.to_matrix()
    # Output storage list for the wavefunction.
    wavefunction = []
    # Output storage list for the times.
    times = []
    # The initial state vector to evolve.
    v_t1 = gs_vec.copy()
    # Run the time evolution.
    for i in range(0, int(t_max / dt) + 1):
        # Must discretize the time evolution.
        t = i * dt
        # Chooses the correct Hamiltonian type (1p, 1m, 2p, 2m)
        if dif == '1m':
            ham1_t = (ham - (amp * float(np.sin(omeg * t)) * dip_z)).reduce() #compute an H(t)
            Tham1_t = qDRIFT(ham1_t, dt, n)
        elif dif == '1p':
            ham1_t = (ham + (amp * float(np.sin(omeg * t)) * dip_z)).reduce() #compute an H(t)
            Tham1_t = qDRIFT(ham1_t, dt, n)
        elif dif == '2m':
            ham1_t = (ham - (2 * amp * float(np.sin(omeg * t)) * dip_z)).reduce() #compute an H(t)
            Tham1_t = qDRIFT(ham1_t, dt, n)
        elif dif == '2p':
            ham1_t = (ham + (2 * amp * float(np.sin(omeg * t)) * dip_z)).reduce() #compute an H(t)
            Tham1_t = qDRIFT(ham1_t, dt, n)
        else:
            raise Exception('Wrong Hamiltonian type. Must be 1p, 1m, 2p, or 2m.')
        # Evolve forward using qDRIFT matrices.
        for op in Tham1_t:
            v_t1 = np.dot(op.to_matrix(), v_t1)
        # Append to storage lists
        wavefunction.append(v_t1)
        times.append(t * conv)
    # Write storage lists out to file.
    with h5py.File(f'./BH_data/wavefunctions_{dif}_{dt*conv}as_{t_max*conv}as-{n}-qDRIFT.hdf5', 'w') as f:
        f.create_dataset(f"mu_{dif}", data=wavefunction)
        f.create_dataset("time", data=times)

    return None

def find_gs_vec(eigvals, eigvecs):
    cgs_energy = min(eigvals)
    for i, _ in enumerate(eigvals):
        if eigvals[i] == cgs_energy:
            gs_vec = eigvecs[:,i]
    return gs_vec
