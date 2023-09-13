#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *
from solvers import run_solver


def run(args):
  ctx = get_context(args)

  ref_gs = run_solver(args, args.Y, ctx)
  print(f'>> ref_gs: {ref_gs:}')
  mes_gs, ansatz = run_solver(args, args.X, ctx)
  print(f'>> mes_gs: {mes_gs:}')

  err = abs(mes_gs - ref_gs)
  print(f'>> Error: {err}')
  err_rate = err / abs(ref_gs)
  print(f'>> Error Rate: {err_rate:%}')

  duration = run_pulse(args, ansatz)
  print(f'>> Duration: {duration}')


if __name__ == '__main__':
  parser = ArgumentParser()
  # hardware
  parser.add_argument('-S', default='Montreal', choices=['Montreal'], help='system model (27 qubits)')
  parser.add_argument('-N', default='', choices=['', 'cairo', 'kolkata', 'montreal'], help='noise model data (27 qubits)')
  # ham
  parser.add_argument('--thresh', default=1e-6, type=float, help='ham term trim amplitude thresh')
  # eigensolver
  parser.add_argument('-Y', default='const', choices=['numpy', 'const'], help='classical eigensolver (ref_val)')
  parser.add_argument('-X', default='vqe', choices=['vqe', 'adavqe', 'svqe', 'qaoa'], help='quantum eigensolver')
  # ansatz
  parser.add_argument
  # optim
  parser.add_argument('-O', default='cobyla', choices=OPTIMZERS.keys(), help='optim method')
  parser.add_argument('-T', '--maxiter', default=10, type=int, help='optim maxiter')
  parser.add_argument('--tol', default=1e-5, type=float, help='optim tolerance')
  parser.add_argument('--disp', action='store_true', help='optim show verbose result')
  # misc
  parser.add_argument('--seed', default=170, type=int, help='rand seed')
  args = parser.parse_args()

  seed_everything(args.seed)

  print(vars(args))
  run(args)
