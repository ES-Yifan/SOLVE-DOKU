import numpy as np
from itertools import product
import time

"""EMUSHI SAKURABA JAN 2026"""


sample_inputs = ['000000602',
                 '057040031',
                 '003000000',
                 '000000007',
                 '100750000',
                 '740010096',
                 '420085060',
                 '001370008',
                 '000900010']

# 0 for unkown


RK = 3   #rank of the sudoku. RK=3 means (3*3) * (3*3) = 9*9



def four_dimensionalise(two_sudoku):
    "rearrange the sudoku as a 4D cube, easier for indexing. Each element is a binary number with RK*RK digits"
    "where n-th digit being 1 means n is possible, being 0 means impossible"
    sudoku = np.array(two_sudoku)
    four_sudoku = np.zeros((RK,RK,RK,RK))
    for n in range(RK):
        for m in range(RK):
            element = sudoku[RK*n:RK*n+RK, RK*m: RK*m+RK]
            four_sudoku[n,m,:,:] = (2**element)/2*(element!=0)+(2**RK**2-1)*(element==0)
    return four_sudoku.astype(np.int32)

def two_dimensionalise(four_sudoku):
    "rearrange the 4D cube back to readable sudoku, for result display"
    two_sudoku = np.zeros((RK**2,RK**2))
    for n in range(RK):
        for m in range(RK):
            two_sudoku[RK*n:RK*n+RK, RK*m: RK*m+RK] = np.log2(four_sudoku[n,m,:,:])+1
    return two_sudoku

def zero_one_combinations(num):
    "generate all binary numbers with length num"
    bin_combinations = np.arange(1,1<<num)
    shifts = np.arange(0,num)
    combinations = (bin_combinations.reshape(-1,1) >> shifts) & 1
    return combinations




def solve(raw_four_sudoku,N=15):
    "solves the sudoku"
    
    sudoku = raw_four_sudoku.copy()

    for n in range(N):
        for (a,b) in product(range(RK),repeat=2):           
            for sliced in (sudoku[a,b,:,:],sudoku[a,:,b,:],sudoku[:,b,:,a]):
                
                #firstly elimates the obvious single values from other blocks
                which_is_definite = (np.bitwise_count(sliced)==1)
                elimated = sliced & ~np.bitwise_or.reduce(sliced[which_is_definite])
                sliced[:,:] = np.where(elimated == 0, sliced, elimated)

                #find which elements remain unknown
                indefinites, ind_counts = np.unique(sliced[~which_is_definite],return_counts=True)

                #find all possible 'unions'; 
                combinations = zero_one_combinations(len(indefinites))
                possible_unions = np.bitwise_or.reduce(combinations*indefinites,axis=1)
                union_involved_counts = np.sum(combinations*ind_counts,axis=1)

                #if no. of unique elements in a union equals no. of blocks that form the union,
                #those elements are occupied (despite unknown order within the union)
                definitely_occupied = possible_unions[union_involved_counts>=np.bitwise_count(possible_unions)]


                #if there is no such union, skip.
                if len(definitely_occupied) == 0:
                    continue

                #if there is at least one such union, eliminate the definitely occupied digit.
                further_elimated = np.bitwise_and.outer(sliced,~definitely_occupied)
                duplicated_sliced = np.bitwise_or.outer(sliced,np.zeros_like(definitely_occupied))
                possible_sliced = np.where(further_elimated==0, duplicated_sliced, further_elimated)

                sliced[:,:] = np.bitwise_and.reduce(possible_sliced, axis = 2)

        if ((sudoku > 0) & (sudoku & (sudoku - 1) == 0) ).all():
            break #check if each element already contains only 1 number of 1.


    return sudoku


if __name__ == "__main__":

    RK = 3

    #str_inputs = [input() for i in range(RK*RK)]
    str_inputs = sample_inputs


    inputs = [[int(string[i]) for i in range(9)] for string in str_inputs]

    print(inputs)



    start = time.time()


    sudoku1 = four_dimensionalise(inputs)

    solution = solve(sudoku1,N=12)

    print(two_dimensionalise(solution))


    end = time.time()
    print(end-start)