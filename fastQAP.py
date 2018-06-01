# -*- coding:utf-8 -*-

import sys
import getopt
import numpy
from scipy import optimize


def normolize_mat(M):
    mmin, mmax = M.min(), M.max()
    return (M-mmin)/(mmax-mmin)


def fun(alpha, A, B, P, Q):
    P_m = P + alpha*Q
    return -numpy.trace(A @ P_m @ B.T @ P_m.T)


def grad(A, B, P):
    return -(A @ P @ B.T) - (A.T @ P @ B)


def hungarian_alg(grad_P):
    # The method used is the Hungarian algorithm,
    # also known as the Munkres or Kuhn-Munkres algorithm.
    print(grad_P)
    print("=========================")
    assign_idx, assign_vec = optimize.linear_sum_assignment(grad_P)
    Q_i_T = numpy.zeros(grad_P.shape)
    for row in assign_idx:
        Q_i_T[row, assign_vec[row]] = 1
    return Q_i_T.T


def min_arg(A, B, P, Q):
    alpha = optimize.fminbound(fun, 0, 1, [A, B, P, Q])
    return alpha


def get_step(P_next, P):
    return numpy.linalg.norm(P_next-P, 'fro')


def permu_projector(P):
    '''
    Maximum is the same as minimum problem
    We find the maximum element m in the matrix P
    then let new P = m * [1] - P
    So far the problem has turned into a minimum assign problem,
    use the same method above.
    '''
    N = P.shape
    P_reverse = numpy.max(P) * numpy.ones(N) - P
    assign_idx, assign_vec = optimize.linear_sum_assignment(P_reverse)
    P_f = numpy.zeros(N)
    for row in assign_idx:
        P_f[row, assign_vec[row]] = 1
    return P_f


def get_QAP_score(A, B, P):
    gap = A - P @ B @ P.T
    return numpy.linalg.norm(gap, 'fro')


def fast_qap(A, B, IMAX, sigma):
    # A, B should be n*n matrix.
    N = A.shape
    P = numpy.ones(N)/N[1]
    i = 0
    step = sigma
    # normolize A & B
    A_norm = normolize_mat(A)
    B_norm = normolize_mat(B)
    while (i < IMAX) and (step >= sigma):
        grad_P = grad(A_norm, B_norm, P)
        Q_i = hungarian_alg(grad_P.T)
        P_next = P + min_arg(A_norm, B_norm, P, Q_i) * Q_i
        step = get_step(P_next, P)
        P = P_next.copy()
        i = i + 1
    result_P = permu_projector(P)
    return get_QAP_score(A_norm, B_norm, result_P)


def main(argv=None):
    inputfile = ""
    try:
        opts, args = getopt.getopt(argv, "i:", ["infile="])
    except getopt.GetoptError:
        print("Error: test_arg.py -i <inputfile>")
        return 2
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            inputfile = arg
    print("Parsing file", inputfile, "...")

    A = numpy.array([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])

    B = numpy.array([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]])

    # C = numpy.random.random((10,10))

    print("score:", fast_qap(A, B, 30, 1.0e-7))
    # print(fast_qap(A, C, 50, 1.0e-7))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
