def compute_data(max_number):
    numbers = [] 
    for i0 in range(max_number):
        for i1 in range(max_number):
            for i2 in range(max_number):
                for i3 in range(max_number):
                    if i0 + i1 + i2 + i3 == max_number and i0 != 0  and i1 != 0  and i2 != 0  and i3 != 0  : 
                              numbers.append([i0 , i1 , i2 , i3]) 
    return numbers