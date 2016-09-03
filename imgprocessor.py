import math


def max_matrix(matrix):
    return max([max(each) for each in matrix])


def min_matrix(matrix):
    return min([min(each) for each in matrix])


def generic_transformation(min_from, max_from, min_to, max_to, v):
    return ((max_to - min_to) / (max_from - min_from)) * \
        (v - max_from) + max_to


def transform_to_std(min_v, max_v, v):
    return generic_transformation(min_v, max_v, 0, 255, v)


def transform_from_std(min_v, max_v, v):
    return generic_transformation(0, 255, min_v, max_v, v)


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

# TODO manage full canvas

class ImageAbstraction:

      # (0,0)      x
      #   | ------ - - - - - - |
      #   |                    |
      #   |                    |
      # y |                    |
      #   |                    |
      #   |                    |
      #   |                    |
      #   | ------ - - - - - - |(w,h)

    def __init__(self, img_list, img_size):
        self.w = img_size[0]
        self.h = img_size[1]
        self.img = ImageAbstraction._get_img_matrix(self.w, self.h, img_list)

    def _get_img_matrix(w, h, img_list):
        img = []
        for i in range(w):
            img.append([])
            for j in range(h):
                img[i].append(img_list[j * w + i])
        return img

    def _get_max_min(self):
        return (max_matrix(self.img), min_matrix(self.img))

    def get_image_list(self):
        pass

    def get_image_bytes(self):
        return bytes(self.get_image_list())

    def get_size_tuple(self):
        return (self.w, self.h)


class BWImageAbstraction(ImageAbstraction):

    def get_image_list(self):
        flat_list = []
        max_v, min_v = self._get_max_min()
        for j in range(self.h):
            for i in range(self.w):
                flat_list.append(
                    int(transform_to_std(min_v, max_v, self.img[i][j])))
        return flat_list

    def update_pixel(self, x, y, color):
        max_v, min_v = self._get_max_min()
        v = transform_from_std(min_v, max_v, color)
        self.img[x][y] = v

    def add(self, image):
        if not self.get_size_tuple() == image.get_size_tuple():
            raise Exception("Not same size")
        self.img = add_matrix(self.img, image.img, self.w, self.h)

    def substract(self, image):
        if not self.get_size_tuple() == image.get_size_tuple():
            raise Exception("Not same size")
        self.img = substract_matrix(self.img, image.img, self.w, self.h)

    def multiply(self, image):
        if not self.h == image.w:
            raise Exception("Not valid for product")
        self.img = multiply_matrix(
            self.img, image.img, self.w, self.h, image.h)

    def negative(self):
        max_v, min_v = self._get_max_min()
        f = lambda x: max_v - x + min_v
        self.img = map_matrix(self.img, self.w, self.h, f)

    def umbral(self, value):
        max_v, min_v = self._get_max_min()
        v = transform_from_std(min_v, max_v, value)
        self.img = map_matrix(
            self.img,
            self.w,
            self.h,
            lambda x: max_v if x > v else min_v)

    def enhance_contrast(self, r1, r2):
        max_v, min_v = self._get_max_min()
        v1 = transform_from_std(min_v, max_v, r1)
        v2 = transform_from_std(min_v, max_v, r2)
        self.img = map_matrix(
            self.img,
            self.w,
            self.h,
            lambda x: 0.5*x if x < v1 else 2*x if x > v2 else x)

    def power(self, value):
        max_v, min_v = self._get_max_min()
        self.img = map_matrix(self.img, self.w, self.h, lambda x: (
            255 / pow(max_v, value)) * pow(x, value))

    def product(self, value):
        self.img = map_matrix(self.img, self.w, self.h, lambda x: x * value)

    def compress(self):
        max_v, min_v = self._get_max_min()
        self.img = map_matrix(self.img, self.w, self.h, lambda x: (
            (255) / (math.log(256))) * math.log(1 + transform_to_std(min_v, max_v, x)))

    def get_mode(self):
        return 'L'
    
    def is_bw(self):
        return True

class RGBImageAbstraction(ImageAbstraction):

    def get_image_list(self):
        flat_list = []
        for j in range(self.h):
            for i in range(self.w):
                flat_list.extend(list(self.img[i][j]))
        return flat_list

    def update_pixel(self, x, y, color):
        raise Exception("Not implemented on RGB")

    def add(self, image):
        raise Exception("Not implemented on RGB")

    def substract(self, image):
        raise Exception("Not implemented on RGB")

    def multiply(self, image):
        raise Exception("Not implemented on RGB")

    def negative(self):
        f = lambda x: tuple(255 - e for e in x)
        self.img = map_matrix(self.img, self.w, self.h, f)

    def get_mode(self):
        return 'RBG'

    def is_bw(self):
        return False

