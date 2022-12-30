# qDRIFT_for_BH

This repository contains files to run the qDRIFT algorithm given a (potentially time-dependent) Hamiltonian. We discretize the time evolution and run qDRIFT on the discretized segments.

There are five files:

BH_ham.txt :: Lists the components of the Hamiltonian operator -- couplings and the operators as Pauli strings. (Couplings in a.u.)

BH_dip.txt :: Lists the components of the dipole operator in the z-direction. (Couplings in a.u.)

utils.py :: Contains the necessary functions.

BH_qDRIFT_script.py :: Runs the algorithm.

BH_run_MPI.py :: Uses MPI (mpi4py) to parallelize the computation.

These scripts output .h5 files that contain the _wavefunctions_ at the desired times. Any observables must be computed from the wavefunctions in post-processing. In the case of the first dipole moment, the equation is:

$<\mu_1> = (8(<\mu_{1-}> - <\mu_{1+}>) - (<\mu_{2-}> - <\mu_{2+}>))/(12 E_0)$
