import math

def euc_dist(pos1, pos2):
    x_diff = pos1[0] - pos2[0]
    y_diff = pos1[1] - pos2[1]

    return math.sqrt(x_diff ** 2 + y_diff ** 2)

def getMatrix(coords):
    N = len(coords)
    matrix = [[0 for i in range(N)] for j in range(N)]

    for i in range(N):
        for j in range(i+1, N):
            dist = euc_dist(coords[i], coords[j])
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix
