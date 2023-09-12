from argparse import ArgumentParser

import numpy as np
from pychemiq import Molecules, ChemiQ, QMachineType
from pychemiq.Transform.Mapping import jordan_wigner, MappingType
from pychemiq.Optimizer import vqe_solver
from pychemiq.Optimizer import (
  DEF_NELDER_MEAD,
  DEF_POWELL,
  DEF_COBYLA,
  DEF_LBFGSB,
  DEF_SLSQP,
  DEF_GRADIENT_DESCENT,
)
from pychemiq.Circuit.Ansatz import UCC

from mol_data import get_mol_data

# VQE with UCCSD/UCCS/UCCD ansatz in pychemiq
#  - https://pychemiq-tutorial.readthedocs.io/en/latest/05theory/vqeintroduction.html


def get_mol() -> Molecules:
  geo, basis, spin, charge = get_mol_data(fmt='chq')
  return Molecules(geo, basis, 2*spin+1, charge)  # 分子体系的自旋多重度 M = 2 * spin + 1


def run_chemiq(args, mol:Molecules):
  fermion = mol.get_molecular_hamiltonian()
  #print(fermion)
  pauli = jordan_wigner(fermion)
  # FIXME: these terms mismatch with `QC-Contest-Demo\Hamiltonian\OHhamiltonian.txt` X(
  #print(pauli)

  machine_type = QMachineType.CPU_SINGLE_THREAD
  mapping_type = MappingType.Jordan_Wigner
  ucc_type = args.type
  n_elec = mol.n_electrons
  n_qubits = mol.n_qubits
  pauli_size = len(pauli.data())
  chemiq = ChemiQ()
  chemiq.prepare_vqe(machine_type, mapping_type, n_elec, pauli_size, n_qubits)
  ansatz = UCC(ucc_type, n_elec, mapping_type, chemiq=chemiq)

  print('ucc_type:', ucc_type)
  print('n_elec:', n_elec)            # 8
  print('n_qubits:', n_qubits)        # 12
  print('pauli_size:', pauli_size)    # 631

  METHODS = [
    DEF_NELDER_MEAD,
    DEF_POWELL,
    DEF_COBYLA,
    DEF_LBFGSB,
    DEF_SLSQP,
    DEF_GRADIENT_DESCENT,
  ]
  for method in METHODS:
    lr = 0.1
    init_para = np.zeros(ansatz.get_para_num()) + 1e-6
    solver = vqe_solver(method, ansatz, pauli, init_para, chemiq, lr)
    result = solver.fun_val
    n_calls = solver.fcalls
    print(f'[{method}] n_calls: {n_calls}, result: {result}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--type', default='UCCSD', choices=['UCCS', 'UCCD', 'UCCSD'], help='ansatze ucc_type')
  args = parser.parse_args()

  mol = get_mol()
  run_chemiq(args, mol)
