A = [[1], [2], [3]]
B = [[1, 0, 1]]
A = [[1,2,3] for i in range(3)]
B = [[1,2,3] for i in range(3)]
C = [[0,0,0] for i in range(3)]

for i in range(len(A)):
    for j in range(len(B)):
        for k in range(len(A)):
            C[i][j] += A[i][k] * B[k][j]

print(C)