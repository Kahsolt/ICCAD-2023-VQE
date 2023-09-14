# ICCAD-2023-VQE

    Quantum Computing for Drug Discovery Challenge at ICCAD 2023

----

Contest page: [https://qccontest.github.io/QC-Contest/](https://qccontest.github.io/QC-Contest/)  


### How to run?

⚪ install

- prepare a working Linux system, [PySCF](https://pyscf.org/) **CANNOT** run on Windows!! :(
- create venv (optional)
  - install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) latest
  - `conda create -n qs`
  - `conda activate qs`
- install dependencies `pip install -r requirements.txt`
- test qiskit installation
  - `python _tutorial_qiskit_vqe.py`
- clone the code & data base `git clone https://github.com/qccontest/QC-Contest-Demo`
- test demo code runnable
  - `python _test_examplecode.py -T 2`; this might be very **slow**, just be patient :(
  - `python _test_NoiseModel_and_SystemModel.py`

⚪ run

- `python run.py`


### references

- Qiskit
  - tutorial: [https://qiskit.org/documentation/tutorials.html](https://qiskit.org/documentation/tutorials.html)
  - doc: [https://qiskit.org/documentation/apidoc/index.html](https://qiskit.org/documentation/apidoc/index.html)
  - algorithms
    - tutorial: [https://qiskit.org/ecosystem/algorithms/tutorials/index.html](https://qiskit.org/ecosystem/algorithms/tutorials/index.html)
    - code: [https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/algorithms](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/algorithms)
- example code: [https://github.com/qccontest/QC-Contest-Demo](https://github.com/qccontest/QC-Contest-Demo)  

----
by Armit
2023/09/05
