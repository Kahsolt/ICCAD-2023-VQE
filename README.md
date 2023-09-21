# ICCAD-2023-VQE

    Quantum Computing for Drug Discovery Challenge at ICCAD 2023

----

Contest page: [https://qccontest.github.io/QC-Contest/](https://qccontest.github.io/QC-Contest/)  


### How to run?

⚪ install

- better prepare a working Linux system, since [PySCF](https://pyscf.org/) **CANNOT** run on Windows!! :(
  - it's also ok to run on Windows, but must use a pre-calculated ham file like [OHhamiltonian.txt](QC-Contest-Demo/Hamiltonian/OHhamiltonian.txt)
  - and few features are unavailable, no big matter
- create venv (optional)
  - install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) latest
  - `conda create -n qs`
  - `conda activate qs`
- install dependencies `pip install -r requirements.txt`
  - if you have CUDA on Linux, one more cmdline: `pip install qiskit-aer-gpu`
- test qiskit installation
  - `python _tutorial_qiskit_vqe.py`
- clone the code & data base `git clone https://github.com/qccontest/QC-Contest-Demo`
- test demo code runnable
  - `python _test_examplecode.py -T 2`; this might be very **slow**, just be patient :(
  - `python _test_NoiseModel_and_SystemModel.py`

⚪ run

⚠ for Windows where PySCF is not available, you could **only** run with a pre-calculated ham file

- `python run.py`, the ham will be computed by Qiskit-PySCF (Linux only!)
- `python run.py -H ham_file` to specify an arbitary pre-calculated ham file
  - run the contest ham: `python run.py -H txt` or `python run.py -H QC-Contest-Demo\Hamiltonian\OHhamiltonian.txt`
  - run the ChemiQ pre-computed ham: `python run.py -H playground\run_chq.ham`
    - this ham seems to be inconsist with the `Qiskit-PySCF` analyzed one, with energy-diff approximately right the value of `nuclear_repulsion_energy`
    - have no idea about know whether it is a bug or feature :(


### references

- Qiskit
  - tutorial: [https://qiskit.org/documentation/tutorials.html](https://qiskit.org/documentation/tutorials.html)
  - doc: [https://qiskit.org/documentation/apidoc/index.html](https://qiskit.org/documentation/apidoc/index.html)
  - algorithms
    - tutorial: [https://qiskit.org/ecosystem/algorithms/tutorials/index.html](https://qiskit.org/ecosystem/algorithms/tutorials/index.html)
    - code: [https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/algorithms](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/algorithms)
- Qiskit ecosystem
  - index: [https://qiskit.org/ecosystem/](https://qiskit.org/ecosystem/)
  - qiskit-aer: [https://github.com/Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)
  - qiskit-nature-pyscf: [https://github.com/qiskit-community/qiskit-nature-pyscf](https://github.com/qiskit-community/qiskit-nature-pyscf)
- base code repo: [https://github.com/qccontest/QC-Contest-Demo](https://github.com/qccontest/QC-Contest-Demo)  
- error mitigation: [https://qiskit.org/ecosystem/ibm-runtime/tutorials/Error-Suppression-and-Error-Mitigation.html](https://qiskit.org/ecosystem/ibm-runtime/tutorials/Error-Suppression-and-Error-Mitigation.html)

----
by Armit
2023/09/05
