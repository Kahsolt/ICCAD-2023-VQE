#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/20

from utils import *
from solvers import solver_const


def eval(args, is_run_pulse:bool=True) -> Tuple[float, int]:
  ctx = get_context(args)
  transpiled_circuit = Circuit.from_qasm_file(args.qsam_fp)
  mapping: QubitMapping = load_json(args.layout_fp)
  mapping = {int(k): v for k, v in mapping.items()}

  ref_val = solver_const(None, None)
  log(f'>> ref_val: {ref_val}')
  ene_gs = run_transpiled_circuit(args, ctx, transpiled_circuit, mapping)
  log(f'>> run qsam result: {ene_gs}, error rate: {abs(abs(ene_gs - ref_val) / ref_val):%}')

  if is_run_pulse:
    system_model = get_backend(args)
    duration = run_pulse(system_model, transpiled_circuit)
    log('>> run pulse duration:', duration)
  else:
    duration = None

  return ene_gs, duration


if __name__ == '__main__':
  args = get_eval_args()
  print(vars(args))
  eval(args)
