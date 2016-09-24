
def max_matrix(matrix):
    return max([max(each) for each in matrix])


def min_matrix(matrix):
    return min([min(each) for each in matrix])


def max_matrix_band(matrix, band):
    return max(max([each[band] for each in e]) for e in matrix)


def min_matrix_band(matrix, band):
    return min(min([each[band] for each in e]) for e in matrix)


def generic_transformation(min_from, max_from, min_to, max_to, v):
    return ((max_to - min_to) / (max_from - min_from)) * \
        (v - max_from) + max_to


def transform_to_std(min_v, max_v, v):
    return generic_transformation(min_v, max_v, 0, 255, v)


def transform_from_std(min_v, max_v, v):
    return generic_transformation(0, 255, min_v, max_v, v)


def sign(x):
    return 1 if x >= 0 else -1


def map_matrix(matrix, w, h, f):
    ret = []
    for i in range(w):
        ret.append([])
        for j in range(h):
            ret[i].append(f(matrix[i][j]))
    return ret


def common_operation_matrix(matrixA, matrixB, w, h, delta):
    ret = [[0 for i in range(h)] for j in range(w)]
    for i in range(w):
        for j in range(h):
            ret[i][j] = matrixA[i][j] + delta * matrixB[i][j]
    return ret


def in_place_multiplication(matrixA, matrixB, w, h):
    ret = [[0 for i in range(h)] for j in range(w)]
    for i in range(w):
        for j in range(h):
            ret[i][j] = matrixA[i][j] * matrixB[i][j]
    return ret


def add_matrix(matrixA, matrixB, w, h):
    return common_operation_matrix(matrixA, matrixB, w, h, 1)


def substract_matrix(matrixA, matrixB, w, h):
    return common_operation_matrix(matrixA, matrixB, w, h, -1)


def multiply_matrix(matrixA, matrixB, w, c, h):
    ret = [[0 for i in range(h)] for j in range(w)]
    for i in range(w):
        for j in range(h):
            aux = 0
            for x in range(c):
                aux += matrixA[i][x] * matrixB[x][j]
            ret[i][j] = aux
    return ret


def flat_matrix(matrix):
    return [item for sublist in matrix for item in sublist]


def flat_img_matrix(matrix, w, h):
    flat_list = []
    for j in range(h):
        for i in range(w):
            flat_list.append(int(matrix[i][j]))
    return flat_list


def put_mask(original, mask, size):
    ret = 0
    for i in range(size):
        for j in range(size):
            ret += original[i][j] * mask[i][j]
    return ret


def get_kirsh_directional_matrix():
    # 5  5  5    5  5 -3  -3 -3 -3   5 -3 -3
    # -3 0 -3    5  0 -3   5  0 -3   5  0 -3
    # -3 -3 -3  -3 -3 -3   5  5 -3   5 -3 -3
    ret = []
    ret.append([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    ret.append([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ret.append([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    ret.append([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    return ret


def get_prewitt_directional_matrix():
    # 1  1  1    1  1  0    0 -1 -1    1  0 -1
    # 0  0  0    1  0 -1    1  0  1    1  0 -1
    # -1 -1 -1   0 -1 -1    1  1  0    1  0 -1
    ret = []
    ret.append([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ret.append([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
    ret.append([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    ret.append([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return ret


def get_sobel_directional_matrix():
    # 1  2  1    2  1  0    0 -1 -2    1  0 -1
    # 0  0  0    1  0 -1    1  0  1    2  0 -2
    # -1 -2 -1   0 -1 -2    2  1  0    1  0 -1
    ret = []
    ret.append([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ret.append([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    ret.append([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    ret.append([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return ret


def get_alternative_directional_matrix():
    # 1  1  1    1  1  1    1 -1 -1   1  1 -1
    # 1 -2  1    1 -2 -1    1 -2 -1   1 -2 -1
    # -1 -1 -1   1 -1 -1    1  1  1   1  1 -1
    ret = []
    ret.append([[1, 1, -1], [1, -2, -1], [1, -1, -1]])
    ret.append([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
    ret.append([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]])
    ret.append([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
    return ret

