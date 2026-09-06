# cython: boundscheck=False, wraparound=False
from libc.stdlib cimport malloc, free
from cython cimport int

cpdef list single_strike_payoff_cal(double strike, str option_type, double option_price, double[:] var_range, int qty):
    cdef Py_ssize_t i, n = var_range.shape[0]
    cdef double var_strike
    cdef double option_price_neg_qty = -option_price * qty
    cdef int payoff
    cdef list payoff_array = []

    if option_type == "CE":
        for i in range(n):
            var_strike = var_range[i]
            if var_strike >= strike:
                payoff = <int>((var_strike - strike) * qty)
            else:
                payoff = <int>(option_price_neg_qty)
            payoff_array.append(payoff)
    else:
        for i in range(n):
            var_strike = var_range[i]
            if var_strike < strike:
                payoff = <int>((strike - var_strike) * qty)
            else:
                payoff = <int>(option_price_neg_qty)
            payoff_array.append(payoff)

    return payoff_array
    