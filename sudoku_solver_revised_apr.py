import numpy as np
from itertools import product
import time

"""EMUSHI SAKURABA JAN 2026, revised in APR 2026"""


sample_inputs_1 = [
    "90e700000g0f0064",
    "d03009ge60002000",
    "b60a00000590f000",
    "0g05a6f07800d039",
    "00a080900e000007",
    "0cd9100b02005840",
    "e7b30000gd00a006",
    "0100000d5000ef00",
    "00g0c0b00672800d",
    "00000000c00430ba",
    "700004000bd06200",
    "300b2700800500ef",
    "0004f8d09ceg1600",
    "g0fd0c0a00007482",
    "c01000002000g300",
    "8020500ga000900c"
]

sample_inputs_2 = ['000000602',
                 '057040031',
                 '003000000',
                 '000000007',
                 '100750000',
                 '740010096',
                 '420085060',
                 '001370008',
                 '000900010']

sample_inputs_3 = ["04G800000B000000",
                 "0060F00007000010",
                 "00020000309A0007",
                 "700B01A05020093C",
                 "G000E0320048D070",
                 "924000000001CE00",
                 "008D001000E02063",
                 "E00007D4000G0800",
                 "000E90010C0000G6",
                 "B600DG0000F90002",
                 "FA000260005E0C00",
                 "085G00C304007000",
                 "009004G0F0600001",
                 "A000000010000005",
                 "0000BA0000040090",
                 "C30006000905008A"]

# 0 for unkown
  #rank of the sudoku. RK=3 means (3*3) * (3*3) = 9*9



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

    #generate all possible 01 combinations of RK*RK-1 digits
    all_01_combinations = [zero_one_combinations(num) for num in np.arange(0,RK*RK)]
    all_try_numbers = (1 << np.arange(0,RK*RK))

    for n in range(N):

        initial_possibilities = np.bitwise_count(sudoku)

        for (a,b) in product(range(RK),repeat=2):           
            for sliced in (sudoku[a,b,:,:],sudoku[a,:,b,:],sudoku[:,b,:,a]):
                
                #firstly elimates the obvious single values from other blocks
                which_is_definite = (np.bitwise_count(sliced)<=1)
                singles = sliced[which_is_definite]
                union_of_singles = np.bitwise_or.reduce(singles)

                if singles.size == RK*RK:
                    continue

                #if those singles contradict themselves, return None
                if singles.size > np.bitwise_count(union_of_singles):
                    print("dead")
                    return None
                
                elimated = sliced & ~union_of_singles

                sliced[:,:] = np.where(which_is_definite, sliced, elimated)


                #find which elements remain unknown
                indefinites, ind_counts = np.unique(sliced[~which_is_definite],return_counts=True)

                #find all possible 'unions'; 
                combinations = all_01_combinations[len(indefinites)]
                possible_unions = np.bitwise_or.reduce(combinations*indefinites,axis=1)
                union_involved_counts = np.sum(combinations*ind_counts,axis=1)

                #if no. of unique elements in a union equals no. of blocks that form the union,
                #those elements are occupied (despite unknown order within the union)
                definitely_occupied = possible_unions[union_involved_counts==np.bitwise_count(possible_unions)]


                #if there is no such union, skip.
                if len(definitely_occupied) == 0:
                    continue

                #if there is at least one such union, eliminate the definitely occupied digit.
                further_elimated = np.bitwise_and.outer(sliced,~definitely_occupied)
                duplicated_sliced = np.bitwise_or.outer(sliced,np.zeros_like(definitely_occupied))
                possible_sliced = np.where(further_elimated==0, duplicated_sliced, further_elimated)

                sliced[:,:] = np.bitwise_and.reduce(possible_sliced, axis = 2)

        posibilities = np.bitwise_count(sudoku)

        if (posibilities == 1).all():
            #the sudoku is already solved if each element already contains only 1 number of 1.
            break

        #brutal traceback for backup if there is no progress
        if (posibilities == initial_possibilities).all():
            
            #find the index of the element with fewest possiblities but not 1.
            posibilities_excl_one = np.where(posibilities==1, RK*RK+1, posibilities)
            min_pos_idx_flat = np.argmin(posibilities_excl_one)
            min_pos_idx = np.unravel_index(min_pos_idx_flat,(RK,RK,RK,RK))

            try_sudoku = sudoku.copy()

            for try_number in all_try_numbers:
                
                if sudoku[min_pos_idx] & try_number:

                    try_sudoku[min_pos_idx] = try_number

                    solution = solve(try_sudoku)

                    if solution is None:
                        continue
                        
                    return solution
                
                continue

    return sudoku


if __name__ == "__main__":

    RK = 4

    #str_inputs = [input() for i in range(RK*RK)]
    
    str_inputs = sample_inputs_3

    inputs = [[int(string[i],RK*RK+1) for i in range(RK*RK)] for string in str_inputs]


    print(*inputs,sep="\n")


    start = time.time()


    sudoku1 = four_dimensionalise(inputs)

    solution = solve(sudoku1,N=12)


    end = time.time()


    print(two_dimensionalise(solution))



    print(end-start)