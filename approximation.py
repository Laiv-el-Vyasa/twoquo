import copy
import random

import numpy as np


def get_approximated_qubos(qubo, single_entry_approx, fixed, approximation_steps):
    approx_qubos = {}
    qubodict = get_sorted_qubodict(qubo, single_entry_approx)
    #print(qubodict)
    size = len(qubodict)
    if single_entry_approx:
        size = np.int(len(qubo) * (len(qubo) + 1) / 2)
        percentage_steps = [(x + 1) / size for x in range(size)]
    else:
        percentage_steps = [((x + 1) / (approximation_steps + 1)) for x in range(approximation_steps)]
    #print('Size: ' + str(size))
    #print(percentage_steps)
    last_approx_number = 0
    for i, percentage_bound in enumerate(percentage_steps):
        if fixed:
            if not single_entry_approx or i < len(qubodict):
                new_qubo, last_approx_number, number_of_approx = approx_fixed_number(copy.deepcopy(qubo),
                                                                                     percentage_bound,
                                                                                     last_approx_number, size,
                                                                                     qubodict)
            else:
                break #end of possible approximations reached
        else:
            new_qubo, last_approx_number, number_of_approx = approx_fixed_values(copy.deepcopy(qubo), percentage_bound,
                                                                                 last_approx_number, size, qubodict)

        approx_qubos[str(i + 1)] = {'qubo': new_qubo, 'approximations': number_of_approx,
                                    'percentage_bound': percentage_bound, 'size': size}
        #print(new_qubo)
        qubo = new_qubo

    return approx_qubos, percentage_steps


def approx_fixed_number(qubo, percentage_bound, last_approx_number, size, qubodict):
    approx_number = int(np.ceil(percentage_bound * size))
    number_of_approx = 0
    for j in range(last_approx_number, approx_number):
        _, (idx, idy) = qubodict[j]
        qubo[idx][idy] = 0
        qubo[idy][idx] = 0
        last_approx_number = approx_number
        number_of_approx += 1
    #print('Anzahl an Approximationen: ' + str(number_of_approx))
    return qubo, last_approx_number, number_of_approx


def approx_fixed_values(qubo, percentage_bound, last_approx_number, size, qubodict):
    lower_bound = percentage_bound * qubodict[size - 1][0]
    #print('Lower bound: ' + str(lower_bound))
    #print('Size: ' + str(size))
    number_of_approx = 0
    for i in range(last_approx_number, size):
        value, (idx, idy) = qubodict[i]
        #print(value)
        if value < lower_bound:
            qubo[idx][idy] = 0
            qubo[idy][idx] = 0
            last_approx_number = i + 1
            number_of_approx += 1
    #print('Anzahl an Approximationen: ' + str(number_of_approx))
    return qubo, last_approx_number, number_of_approx


def get_sorted_qubodict(qubo, single_entry_approx):
    dict_list = []
    shape = len(qubo)
    for i in range(shape):
        for j in range(i + 1):
            if not single_entry_approx or not qubo[i][j] == 0:
                dict_list.append((np.absolute(qubo[i][j]), (i, j)))
    random.shuffle(dict_list)
    return sorted(dict_list, key=get_qubo_position_value)


def get_qubo_position_value(dict):
    return dict[0]


def get_max_from_qubodict(qubodict):
    return qubodict[len(qubodict) - 1]


def get_min_position(qubo):
    map(lambda x: np.absolute(x), qubo)
    min_value = get_min_value(qubo)
    pos = np.where(qubo == min_value)
    return pos


def get_min_value(qubo):
    flat_qubo = map(lambda x: np.absolute(x), np.reshape(qubo, -1))
    return min(a for a in flat_qubo if a > 0)
