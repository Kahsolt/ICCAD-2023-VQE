# ICCAD-2023-VQE

    Quantum Computing for Drug Discovery Challenge at ICCAD 2023

----

Contest page: [https://qccontest.github.io/QC-Contest/](https://qccontest.github.io/QC-Contest/)  


### Experimental results

reference cfg: numpy  
baseline quant cfg: vqe-uccsd-cobyla(10)  

⚪ baseline

| method | energy | erorr | duration |
| :-: | :-: | :-: | :-: |
| reference | -74.3871462681872  |  |  |
| baseline | -73.95153349095546 | 0.585602% | 20478368 |

⚪ fewer shots

| shots | energy | erorr | duration |
| :-: | :-: | :-: | :-: |
| 6000 | -73.95153349095546 | 0.585602% | 20478368 |
| 3000 | -73.98363845753188 | 0.542443% | 18088448 |
| 1000 | -73.97834650827683 | 0.549557% | 20428640 |
|  300 | -73.93227485350587 | 0.611492% | 21548736 |
|  100 | -73.8960979814249  | 0.660125% | 19724704 |
|   10 | -74.05916893056036 | **0.440906%** | 21440832 |

⚪ ham trim

| thresh | terms| energy | erorr | duration |
| :-: | :-: | :-: | :-: |
| -     | 631 | -73.95153349095546 | 0.585602% | 20478368 |
| 0.001 | 615 | -73.9467206286837  | 0.592074% | 19945568 |
| 0.01  | 307 | -74.00739269969118 | **0.510831%** | 16972864 |
| 0.1   |  95 | -74.18871758139487 | 0.948692% | 17815648 |
| 0.3   |  14 | -75.05707730713337 | 9.750768% | 21573344 |
| 0.3 (1000 steps) | 14 | -75.08620960765452 | 9.715739% | 20532032 |

⚪ ansatz

| ansatz | energy | erorr | duration |
| :-: | :-: | :-: | :-: |
| uccs   | -73.95002471896069 | 0.587630% | 177056 |
| uccd   | -73.89877488212468 | 0.656527% | 19186784 |
| uccsd  | -73.95153349095546 | **0.585602%** | 18077952 |
| uccst  | -73.95153349095546 | 0.585602% | 118116160 |
| uccdt  | -73.89877488212468 | 0.656527% | 145150272 |
| uccsdt | -73.95153349095546 | 0.585602% | 142490016 |
| puccd  | -73.89725545197842 | 0.658569% | 1801344 |
| succd  | -73.89725545197842 | 0.658569% | 9886560 |
| chcs   | -73.90800071336932 | 0.644124% | **12608** |
| chcd   | -73.90909036362392 | 0.642659% | 129792 |
| chcsd  | -73.90909036362392 | 0.642659% | 147072 |

⚪ init

| init | energy | erorr | duration |
| :-: | :-: | :-: | :-: |
| hf    | -73.95153349095546 | 0.585602% |  |
| none  | -67.48181953651472 | 9.282957% |  |
| zeros | -73.95153349095546 | 0.585602% |  |
| randu | -53.40543284234816 | 28.206101% |  |
| randn | -63.60381674415347 | 14.496227% |  |


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

----
by Armit
2023/09/05
