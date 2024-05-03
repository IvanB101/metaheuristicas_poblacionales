import numpy as np

parent_1 = np.array([1, 2, 5, 6])
parent_2 = np.array([5, 6, 7, 8])

chromosome_length = len(parent_1)


first = parent_1[:2]
last = parent_1[2:]

print(list(set(last) & set(parent_2)))
print(list(set(parent_2) & set(last)))
