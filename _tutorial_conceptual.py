#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/19

# 来不及写科普 ppt 了，写个脚本来快速介绍 VQE 的工作原理
# 忽略了所有量子的部分 :(

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Ham = NDArray[np.float32]


def make_ham(n_qubit=2):
  ''' VQE 能求解的矩阵，要求是 Hermitian 的 '''
  dim = 2 ** n_qubit
  x = np.random.normal(size=[dim, dim])
  x = (x + x.T) / 2
  return x


def solve_linalg(ham:Ham) -> float:
  result = np.linalg.eig(ham)
  return sorted(result.eigenvalues)[0]


def solve_optim(ham:Ham) -> float:
  def func(x) -> float:
    # Rayleigh-Ritz quotient: <x|H|x> / <x|x>
    return np.dot(np.dot(x, ham), x.T) / np.dot(x, x.T)

  x0 = np.random.normal(size=[ham.shape[0]])
  res = minimize(func, x0, tol=1e-6, options={'maxiter':100, 'disp':True})
  return res.fun


def solve_variational(ham:Ham) -> float:
  def func(x) -> float:
    # variational Rayleigh-Ritz quotient: <U(x)|H|U(x)> / <U(x)|U(x)>
    Ux = ansatz(x)
    return np.dot(np.dot(Ux, ham), Ux.T) / np.dot(Ux, Ux.T)

  ansatz = lambda x: np.exp(np.log(np.abs(np.sin(x) + x)) - np.cos(x))
  x0 = np.random.normal(size=[ham.shape[0]])
  res = minimize(func, x0, tol=1e-6, options={'maxiter':100, 'disp':True})
  return res.fun


if __name__ == '__main__':
  ham = make_ham(2)
  print('ham:')
  print(ham)
  print('-' * 42)

  r1 = solve_linalg(ham)
  r2 = solve_optim(ham)
  r3 = solve_variational(ham)
  print('linalg:', r1)
  print('optim:', r2, 'error:', np.abs(r2 - r1))
  print('variational:', r3, 'error:', np.abs(r3 - r1))
