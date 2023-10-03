#!/usr/bin/env python

import argparse

import numpy as np
from typing import List, Tuple, Union, Optional, Any
from scipy.optimize import linprog


def parse_command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read matrix A, vectors b and c from the command line.")

    parser.add_argument("-A", "--matrix_A", type=str, required=True,
                        help="Matrix A in the format '1 2 3; 4 5 6; 7 8 9'")
    parser.add_argument("-b", "--vector_b", type=str,
                        required=True, help="Vector b in the format '1 2 3'")
    parser.add_argument("-c", "--vector_c", type=str,
                        required=True, help="Vector c in the format '1 2 3'")
    parser.add_argument("--checker", help="Check output with scipy",
                        action="store_true", required=False)
    parser.add_argument("-v", "--verbose", help="print steps",
                        action="store_true", required=False)
    args = parser.parse_args()
    return args


# TODO: when 0... is not a BFS
def solve(A: np.ndarray, b: np.ndarray, c: np.ndarray, verbose: bool = False) -> \
        tuple[None, None] | tuple[list[Any], np.ndarray[Any, Any]]:
    T, m, n = construct_simplex_tableau(A, b, c)

    log(verbose, '\n', T, '\n')

    tolerance = 1e-9
    curr_iter = 0
    while True:
        curr_iter = curr_iter + 1
        T_z = T[n - 1][0:m - 1]

        has_negative_z = any(
            map(lambda x: True if x < -tolerance else False, T_z))
        if not has_negative_z:
            break

        if curr_iter >= 10000:
            log(verbose, "Max iterations reached")
            return (None, None)

        pivot_col_i = find_enter_variable(T_z)
        pivot_row_i = find_exit_variable(T, pivot_col_i, b)

        if pivot_row_i is None:
            log(verbose, "No Feasible Solution")
            return (None, None)

        pivot_elem = T[pivot_row_i][pivot_col_i]
        assert pivot_elem != 0
        T[pivot_row_i] = T[pivot_row_i] / pivot_elem

        log(verbose, "Pivot element (row, col): " + str((pivot_row_i, pivot_col_i)))
        log(verbose, 'R_' + str(pivot_row_i) + ' / ' + str(pivot_elem) + ' -> R_' + str(pivot_row_i))

        for i, row in enumerate(T):
            if i == pivot_row_i or abs(T[i][pivot_col_i]) <= tolerance:
                continue
            else:
                log(verbose, '-R_' + str(pivot_row_i)
                    + ' * ' + str(T[i][pivot_col_i])
                    + ' + ' + 'R_' + str(i) + ' -> R_' + str(i))

                T[i] = T[i] - (T[pivot_row_i] * T[i][pivot_col_i])

        log(verbose, '\n', T, '\n')


    result = np.zeros(len(c), dtype=float)
    for j in range(len(c)):
        (is_basic, row_i_one) = is_basic_variable(T, j)
        if is_basic:
            result[j] = T[row_i_one][-1] if abs(T[row_i_one]
                                                [-1]) > tolerance else 0.0

    return (list(result), T[-1][-1])


def log(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def construct_simplex_tableau(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, int, int]:
    n = len(A) + 1
    m = len(A[0]) + len(b) + 1
    T = np.zeros((n, m), dtype=float)
    for i in range(len(A)):
        for j in range(len(A[0])):
            T[i][j] = A[i][j]
    for i in range(len(A[0])):
        T[i][len(A[0]) + i] = 1
    for i in range(len(b)):
        T[i][len(A[0]) + len(b)] = b[i]
    for i in range(len(c)):
        T[len(A)][i] = -c[i]
    return T, m, n


def find_enter_variable(T_z: np.ndarray) -> Optional[int]:
    def get_max_abs(x):
        return abs(x) if x < 0 else 0

    abs_values = map(get_max_abs, T_z)
    enumerated_values = list((v, i) for i, v in enumerate(abs_values))

    pivot_col_i = None
    for value, index in sorted(enumerated_values, key=lambda x: (-x[0], x[1])):
        if value > 0:
            pivot_col_i = index
            break

    return pivot_col_i


def find_exit_variable(T: np.ndarray, pivot_col_i: int, b: np.ndarray, tolerance: float = 1e-9) -> Optional[int]:
    curr_min = None
    pivot_row_i = None
    for i in range(len(b)):
        rhs = T[i][-1]
        elem = T[i][pivot_col_i]
        if elem <= tolerance:
            continue
        ratio = rhs / elem
        if (pivot_row_i is None) or (curr_min is None) or (ratio <= curr_min):
            curr_min = ratio
            pivot_row_i = i
    return pivot_row_i


def is_basic_variable(T: np.ndarray, j: int) -> Tuple[bool, Optional[int]]:
    num_zeros = 0
    num_ones = 0
    row_i_one = None
    for i in range(len(T)):
        if T[i][j] == 0:
            num_zeros = num_zeros + 1
        if T[i][j] == 1:
            num_ones = num_ones + 1
            row_i_one = i

    return (True, row_i_one) if (num_zeros == len(T) - 1) and (num_ones == 1) else (False, None)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    args = parse_command_line_args()
    A = np.array([list(map(float, row.split()))
                  for row in args.matrix_A.strip().split(';')])
    b = np.array(list(map(float, args.vector_b.strip().split())))
    c = np.array(list(map(float, args.vector_c.strip().split())))
    sol = solve(A, b, c, args.verbose)
    print(sol)

    if args.checker:
        expected = linprog(-c, A_ub=A, b_ub=b)
        print((list(expected.x), -expected.fun))
