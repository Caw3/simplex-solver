import unittest
import numpy as np
from simplex_solver import solve
from scipy.optimize import linprog

class TestLP(unittest.TestCase):
    def assertLP(self, A, b, c):
        print(
            '\n---------------------------------------------------------------------------')
        result = solve(A, b, c, verbose=True)
        expected = linprog(-c, A_ub=A, b_ub=b)
        self.assertEqual(result[1], -expected.fun)
        print(result)
        print(list(expected.x), -expected.fun)
        self.assertEqual(np.sum(c * result[0]), result[1])
        self.assertEqual(np.sum(c * result[0]), -expected.fun)

    def test_sanity(self):
        A = np.array([[4, 2], [2, 3]])
        b = np.array([32, 24])
        c = np.array([5, 4])
        self.assertLP(A, b, c)

    def test_sanity_1(self):
        A = np.array([[1, 1], [1, 2], [4, 3]])
        b = np.array([10, 15, 38])
        c = np.array([750, 1000])
        self.assertLP(A, b, c)

    def test_multiple_sol(self):
        A = np.array([[1, 1, 0], [0, 1, 3], [2, 5, 9]])
        b = np.array([2, 3, 10])
        c = np.array([1, 2, 3])
        self.assertLP(A, b, c)

    def test_no_init_bfs(self):
        A = np.array([[-1, -2], [-1, 2], [-1, 2]])
        b = np.array([-3, -1, 2])
        c = np.array([1, 3])
        res = solve(A, b, c, verbose=True)
        self.assertEqual(res[0], None)
        self.assertEqual(res[1], None)

    # def test_unbounded_solution(self):
    #     A = np.array([[1, 1], [-1, 0], [0, -1]])
    #     b = np.array([2, 0, 0])
    #     c = np.array([1, 1])
    #     res = solve(A, b, c, verbose=True)
    #     self.assertEqual(res[0], None)  # No bounded solution for maximization
    #     self.assertEqual(res[1], None)  # Optimal value should be infinite

    # def test_single_constraint(self):
    #     A = np.array([[1, 2]])
    #     b = np.array([4])
    #     c = np.array([2, 1])
    #     self.assertLP(A, b, c)

    # def test_infeasible(self):
    #     A = np.array([[1, 1], [-1, -1], [1, -1]])
    #     b = np.array([1, -2, 0])
    #     c = np.array([1, 1])
    #     res = solve(A, b, c, verbose=True)
    #     self.assertEqual(res[0], None)
    #     self.assertEqual(res[1], None)

    def test_trivial(self):
        A = np.array([[1, 0], [0, 1]])
        b = np.array([1, 1])
        c = np.array([1, 1])
        self.assertLP(A, b, c)

    def test_tight_constraints(self):
        A = np.array([[1, 1], [1, -1]])
        b = np.array([1, 0])
        c = np.array([1, 1])
        self.assertLP(A, b, c)

    def test_zero_objective(self):
        A = np.array([[1, 2], [2, 3]])
        b = np.array([4, 7])
        c = np.array([0, 0])
        self.assertLP(A, b, c)

    # def test_negative_coefficients(self):
    #     A = np.array([[-1, 2], [2, -3]])
    #     b = np.array([-1, 4])
    #     c = np.array([-1, -2])
    #     self.assertLP(A, b, c)



def run_tests():
    unittest.main()