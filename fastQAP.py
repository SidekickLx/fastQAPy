import sys
import numpy
from scipy import optimize

def fun(alpha, A, B, P, Q) :
    P_m = P + alpha*Q
    return numpy.trace(A @ P_m @ B.T @ P_m.T)

def grad(A, B, P) :
    return (A @ P @ B.T) - (A.T @ P @ B)

def hungarian_alg(grad_P) :
    #The method used is the Hungarian algorithm, also known as the Munkres or Kuhn-Munkres algorithm.
    assign_idx, assign_vec = optimize.linear_sum_assignment(grad_P) 
    Q_i_T = numpy.zeros(grad_P.shape)
    for row in assign_idx:
        Q_i_T[row, assign_vec[row]] = 1
    return Q_i_T.T

def min_arg(A, B, P, Q) :
    #alpha = 1
    alpha = optimize.fminbound(fun, 0, 1, [A, B, P, Q])
    return alpha

def get_step(P_next, P):
    return numpy.linalg.norm(P_next-P, 'fro')

def permu_projector(P) :
    #最大化问题与最小化问题其实是一样的
    #找到系数矩阵中最大的元素，然后使用最大元素分别减去矩阵中元素，这样最大化问题就化为最小化问题，同样可以使用匈牙利算法。
    N= P.shape
    P_reverse = numpy.max(P) * numpy.ones(N) - P
    assign_idx, assign_vec = optimize.linear_sum_assignment(P_reverse)
    P_f = numpy.zeros(N)
    for row in assign_idx:
        P_f[row, assign_vec[row]] = 1
    return P_f

def get_QAP_score(A, B, P) :
    gap = A - P @ B @ P.T
    return numpy.linalg.norm(gap, 'fro')

def fast_qap(A, B, IMAX, sigma) :
    #A, B should be n*n matrix.
    N = A.shape
    P = numpy.ones(N)/N[1]
    i = 0
    step = sigma
    while (i <  IMAX) and (step >= sigma) :
        grad_P = grad(A, B, P)
        Q_i = hungarian_alg(grad_P.T)
        P_next = P + min_arg(A, B, P, Q_i) * Q_i
        step = get_step(P_next, P)
        P = P_next.copy()
        i = i + 1
    result_P  = permu_projector(P)
    return get_QAP_score(A, B, result_P)


def  main(argv=None) :
    if argv is None:
        argv = sys.argv
    A = numpy.random.randn(10,10)
    B = numpy.random.randn(10,10)

    print(A)
    print(B)
    print(fast_qap(A, B, 10, 1.0e-4))

if __name__ == "__main__":
   sys.exit(main())