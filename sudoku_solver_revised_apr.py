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

sample_inputs_4 = [
    "000056700000000000J0LM00P",
    "00090BC0E00H0J0L000000000",
    "000E000I00L00OP100000780A",
    "0H000000O002000008000C0EF",
    "0MN00000056000AB0000G0I00",
    "00456009A000EF0000K000O00",
    "7000000000000KL00O0020056",
    "00000HIJ000N00000050700AB",
    "HIJ00000010300670900C00F0",
    "00O010005008000000F0H00K0",
    "040078000CDE00HI0KL0N0P10",
    "00000D0000IJ0000O00200007",
    "D0F0000KLMN00023000789AB0",
    "I0KLMN0P023050080000DE000",
    "N0P02040008000000F0H0JKL0",
    "0500890000000HIJ000000020",
    "000CD00000JKL000002000070",
    "0F0HI000M000103450780A0C0",
    "JK0000P0000000000BC000000",
    "O01230007890B0000000J0000",
    "000000000000H0JK0000P0234",
    "AB0000G00JK00N0P100056080",
    "F00IJ00MN00000450709A0CD0",
    "K000O0120400000A00D0FG00J",
    "0103400700A00000G0I000000",
]

# 0 for unkown
  #rank of the sudoku. RK=3 means (3*3) * (3*3) = 9*9



def four_dimensionalise(two_sudoku):
    "rearrange the sudoku as a 4D cube, easier for indexing. Each element is a binary number with RK*RK digits"
    "where n-th digit being 1 means n is possible, being 0 means impossible"

    sudoku = np.array(two_sudoku)
    RK = int(sudoku.shape[0]**(1/2))

    four_sudoku = np.zeros((RK,RK,RK,RK))
    for n in range(RK):
        for m in range(RK):
            element = sudoku[RK*n:RK*n+RK, RK*m: RK*m+RK]
            four_sudoku[n,m,:,:] = (2**element)/2*(element!=0)+(2**RK**2-1)*(element==0)
    return four_sudoku.astype(np.int32)

def two_dimensionalise(four_sudoku):
    "rearrange the 4D cube back to readable sudoku, for result display"
    RK = four_sudoku.shape[0]
    if four_sudoku is None:
        return np.zeros((RK**2,RK**2))
    two_sudoku = np.zeros((RK**2,RK**2))
    for n in range(RK):
        for m in range(RK):
            two_sudoku[RK*n:RK*n+RK, RK*m: RK*m+RK] = np.where(np.bitwise_count(four_sudoku[n,m,:,:]) == 1, np.log2(four_sudoku[n,m,:,:])+1,0.)

    return two_sudoku

def zero_one_combinations(num):
    "generate all binary numbers with length num"
    bin_combinations = np.arange(1,1<<num)
    shifts = np.arange(0,num)
    combinations = (bin_combinations.reshape(-1,1) >> shifts) & 1
    return combinations



def solve(raw_four_sudoku,max_cycle=300000,economic_skip=5):

    RK = raw_four_sudoku.shape[0]

    #generate all possible 01 combinations of RK*RK-1 digits.
    all_01_combinations = [zero_one_combinations(num) for num in np.arange(0,RK*2+economic_skip+1)]
    all_try_numbers = (1 << np.arange(0,RK*RK))

    def solve0(raw_four_sudoku):
        "solves the sudoku"
        
        sudoku = raw_four_sudoku.copy()

        for n in range(max_cycle):
            initial_possibilities = np.bitwise_count(sudoku)

            for (a,b) in product(range(RK),repeat=2):           
                for sliced in (sudoku[a,b,:,:],sudoku[a,:,b,:],sudoku[:,b,:,a]):
                    
                    #firstly elimates the obvious single values from other blocks
                    which_is_definite = (np.bitwise_count(sliced)<=1)
                    singles = sliced[which_is_definite]
                    union_of_singles = np.bitwise_or.reduce(singles)

                    # if this sliced sudoku is already solved
                    if singles.size == RK*RK:
                        continue

                    #if those singles contradict themselves, return None
                    if singles.size > np.bitwise_count(union_of_singles):
                        return None
                    
                    elimated = sliced & ~union_of_singles

                    sliced[:,:] = np.where(which_is_definite, sliced, elimated)


                    #find which elements remain unknown
                    indefinites, ind_counts = np.unique(sliced[~which_is_definite],return_counts=True)

                    #give up when it is too hard and not economical to determine
                    if len(indefinites) > RK*2+economic_skip:
                        continue

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


            if (posibilities != initial_possibilities).any():
                
                continue


            #brutal traceback for backup if there is no progress
            #find the index of the element with fewest possiblities but not 1.
            posibilities_excl_one = np.where(posibilities==1, RK*RK+1, posibilities)
            min_pos_idx_flat = np.argmin(posibilities_excl_one)
            min_pos_idx = np.unravel_index(min_pos_idx_flat,(RK,RK,RK,RK))

            original_pos = sudoku[min_pos_idx] 

            for try_number in (original_pos & all_try_numbers):
                
                if try_number:

                    sudoku[min_pos_idx] = try_number

                    solution = solve0(sudoku)

                    if solution is None:
                        continue

                    return solution
                
                continue

            return None
        
        return sudoku
    
    return solve0(raw_four_sudoku)


if __name__ == "__main__":


    #str_inputs = [input() for i in range(RK*RK)]
    
    str_inputs = sample_inputs_3

    inputs = [[int(string[i],36) for i in range(len(string))] for string in str_inputs]


    print(*inputs,sep="\n")


    start = time.time()


    sudoku1 = four_dimensionalise(inputs)

    solution = solve(sudoku1)


    end = time.time()


    print(two_dimensionalise(solution))



    print(end-start)
