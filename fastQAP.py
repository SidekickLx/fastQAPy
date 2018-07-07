# -*- coding:utf-8 -*-

import sys
import random
import numpy
from scipy import optimize
# import getopt
# from graphAnalyzer import graphAna
# import time
# import matplotlib.pyplot as plt


def normolize_mat(M):
    mmin, mmax = M.min(), M.max()
    return (M-mmin)/(mmax-mmin)


def fun(alpha, A, B, P, Q, D):
    P_m = P + alpha*Q
    return -numpy.trace(A @ P_m @ B.T @ P_m.T)+numpy.trace(D @ P_m)


def grad(A, B, P, D):
    return -(A @ P @ B.T) - (A.T @ P @ B) + D.T


def hungarian_alg(grad_P):
    '''
    The hungarian algorithm is implemented by linear_sum_assignment
    from scipy.optimize module
    '''
    # The method used Hungarian algorithm,
    # also known as the Munkres or Kuhn-Munkres algorithm.
    assign_idx, assign_vec = optimize.linear_sum_assignment(grad_P)
    Q_i_T = numpy.zeros(grad_P.shape)
    for row in assign_idx:
        Q_i_T[row, assign_vec[row]] = 1
    return Q_i_T.T


def min_arg(A, B, P, Q, D):
    alpha = optimize.fminbound(fun, 0, 1, [A, B, P, Q, D])
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


def get_QAP_score(A, B, P, D):
    gap = A - P @ B @ P.T
    return numpy.linalg.norm(gap, 'fro') + numpy.trace(D @ P)


def fast_qap(A, B, D, IMAX, sigma):
    # A, B should be n*n matrix.
    N = A.shape
    P = numpy.ones(N)/N[1]
    i = 0
    step = sigma
    # normolize A & B
    A_norm = normolize_mat(A)
    B_norm = normolize_mat(B)
    while (i < IMAX) and (step >= sigma):
        grad_P = grad(A_norm, B_norm, P, D)
        Q_i = hungarian_alg(grad_P.T)  # where I made a modification
        # Q_i = Q_i - P                # The paper might make a mistake here
        P_next = P + min_arg(A_norm, B_norm, P, Q_i, D) * Q_i
        step = get_step(P_next, P)
        P = P_next.copy()
        i = i + 1
    result_P = permu_projector(P)
    return get_QAP_score(A_norm, B_norm, result_P, D), result_P


def main(argv=None):
    # try:
    #     opts, args = getopt.getopt(argv, "i1:i2:", ["infile1=", "infile2="])
    # except getopt.GetoptError:
    #     print("Error: test_arg.py -i <inputfile>")
    #     return 2
    # for opt, arg in opts:
    #     if opt in ("-i1", "--infile1"):
    #         inputfile1 = arg
    #     if opt in ("-i2", "--infile2"):
    #         inputfile2 = arg

    # A = graphAna.CFG(inputfile1)
    # B = graphAna.CFG(inputfile2)

    # diff_map = graphAna.analysor(A.node_stmt_list, B.node_stmt_list)

    numpy.set_printoptions(threshold=numpy.inf)  # set output format

    size = 400
    A = numpy.zeros(size*size)
    a = [random.randint(0, size*size)
         for __ in range(int(size/4))]
    for idx in a:
        A[idx] = 1
    A = A.reshape([size, size])
    D = normolize_mat(numpy.random.random([size, size]))
    p = numpy.arange(size)
    random.shuffle(p)
    P = numpy.zeros([size, size])
    for idx in range(size):
        P[idx][p[idx]] = 1
    B = P @ A @ P.T
    # start = time.clock()
    # Do with diff_map
    score, result = fast_qap(A, B, D, 60, 1e-7)
    # elapsed = (time.clock() - start)

    D_zero = numpy.zeros([size, size])
    score2, result2 = fast_qap(A, B, D_zero, 60, 1e7)
    score2 = score2 + numpy.trace(D @ P)

    print("with diff_map: ", score, "D*P: ", numpy.trace(D @ result))
    # print(result)
    print("without diff_map: ", score2, "D*P: ",  numpy.trace(D @ result2))
    # print(result2)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
