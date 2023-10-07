#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/08

from re import compile as Regex
from typing import List, Tuple
from math import pi

Lines = List[str]
Gate = Tuple[str, str]
Qubit = str

''' helpers '''

PGATES = ['rx', 'ry', 'rz']
R_LINE = Regex('(.+) (.+);')
R_PGATE = Regex('(\w+)\((.+)\)')

def _split_gate_qubit(line:str) -> Tuple[Gate, Qubit]:
  gate_param, qubit = R_LINE.findall(line)[0]
  m = R_PGATE.findall(gate_param)
  if m: gate, param = m[0]
  else: gate, param = gate_param, None
  return (gate, param), qubit

EPS = 1e-6
PHI_ZERO = '0'
N_DIV = 32
PI_TABLE = { pi: 'pi' }
PI_TABLE.update({  pi / n:  f'pi/{n}' for n in range(2, N_DIV+1)   })
PI_TABLE.update({ -pi / n: f'-pi/{n}' for n in range(N_DIV, 1, -1) })

def _try_rnd_to_pi(phi:float) -> str:
  if abs(phi) < EPS: return PHI_ZERO

  val = phi
  for k, v in PI_TABLE.items():
    if abs(k - phi) < EPS:
      val = v
      break
  return val

def _norm_rot_angle(phi:float) -> float:
  phi %= 2 * pi                 # vrng in (-2*pi, 2*pi)
  if phi > +pi: phi -= 2 * pi   # (pi, 2*pi) -> # (-pi, 0)
  if phi < -pi: phi += 2 * pi   # (-2*pi, -pi) -> # (0, pi)
  return phi                    # vrng in (-pi, pi)


''' processors '''

def fold_rot_gate(lines:Lines) -> Lines:
  ret = [lines[0]]

  for this in lines[1:]:
    (g0, p0), q0 = _split_gate_qubit(this)
    (g1, p1), q1 = _split_gate_qubit(ret[-1])
    if q0 == q1 and g0 == g1 and g0 in PGATES:
      phi: float = eval(p0) + eval(p1)
      phi = _norm_rot_angle(phi)
      phi_str: str = _try_rnd_to_pi(phi)
      ret.pop(-1)
      if phi_str != PHI_ZERO:
        ret.append(f'{g0}({phi_str}) {q0};')
    else:
      ret.append(this)

  return ret

def sanitize_rot_gate_angle(lines:Lines) -> Lines:
  ret = []

  for this in lines:
    (g0, p0), q0 = _split_gate_qubit(this)
    if g0 in PGATES:
      phi: float = eval(p0)
      phi = _norm_rot_angle(phi)
      phi_str: str = _try_rnd_to_pi(phi)
      if phi_str != PHI_ZERO:
        ret.append(f'{g0}({phi_str}) {q0};')
    else:
      ret.append(this)

  return ret


''' entry '''

def optimize(qasm:str) -> str:
  def _split(qsam:str) -> Tuple[Lines, Lines]:
    lines = qsam.strip().split('\n')
    return lines[:2], lines[2:]
  def _combine(head:Lines, body:Lines) -> str:
    return '\n'.join(head + body)

  head, body = _split(qasm)
  len_raw = len(body)
  body = fold_rot_gate(body)
  body = sanitize_rot_gate_angle(body)
  len_opt = len(body)
  print(f'>> qasm_optimize: {len_raw} => {len_opt}')
  return _combine(head, body)


if __name__ == '__main__':
  qasm = '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
x q[0];
rz(0.7949563763792666) q[0];
sx q[0];
rz(pi) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(-0.7758399504156299) q[1];
sx q[1];
rz(pi) q[1];
sx q[1];
rz(3*pi) q[1];
x q[2];
'''

  qasm_opt = optimize(qasm)
  print(qasm_opt)
