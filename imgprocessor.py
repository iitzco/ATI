import math
import random
from collections import Counter
from statistics import median


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


def flat_matrix(matrix):
    return [item for sublist in matrix for item in sublist]

def flat_img_matrix(matrix, w, h):
    flat_list = []
    for j in range(h):
        for i in range(w):
            flat_list.append(
                int(matrix[i][j]))
    return flat_list

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

    def get_mode(self):
        return 'L'

    def is_bw(self):
        return True

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

    def get_pixel_color(self, x, y):
        max_v, min_v = self._get_max_min()
        c = int(transform_to_std(min_v, max_v, self.img[x][y]))
        return (c, c, c)

    def get_bw_img(self):
        return self

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
            lambda x: 0.5 * x if x < v1 else 2 * x if x > v2 else x)

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

    def equalize(self):
        normalized_img = self.get_image_list()
        aux_matrix = ImageAbstraction._get_img_matrix(
            self.w, self.h, normalized_img)
        c = Counter(normalized_img)
        total = self.w * self.h
        s_list = [0] * 256
        s_list[255] = total
        for i in range(1, 256):
            s_list[255 - i] = s_list[256 - i] - c[256 - i]
        s_list = [each / total for each in s_list]
        min_value = min(s_list)
        self.img = map_matrix(aux_matrix, self.w, self.h, lambda x: (
            int(((s_list[x] - min_value) / (1 - min_value)) * 255 + 0.5)))

    def contaminate_multiplicative_noise(self, percentage, generator):
        total = self.w * self.h
        pixels = int((total * percentage) / 100)
        candidates = random.sample(range(total), pixels)
        for each in candidates:
            noise = generator(random.random())
            x = each // self.h
            y = each % self.h
            self.img[x][y] = self.img[x][y] * noise

    def contaminate_gauss_noise(self, percentage, intensity):
        normalized_img = self.get_image_list()
        aux_matrix = ImageAbstraction._get_img_matrix(
            self.w, self.h, normalized_img)
        total = self.w * self.h
        pixels = int((total * percentage) / 100)
        noise_list = []
        for i in range((pixels + 1) // 2):
            x1, x2 = random.random(), random.random()
            noise_list.append(math.sqrt(-2 * math.log(x1))
                              * math.cos(2 * math.pi * x2))
            noise_list.append(math.sqrt(-2 * math.log(x1))
                              * math.sin(2 * math.pi * x2))
        candidates = random.sample(range(total), pixels)
        for each in candidates:
            x = each // self.h
            y = each % self.h
            self.img[x][y] = aux_matrix[x][y] + \
                (noise_list.pop() * intensity)

    def contaminate_salt_pepper_noise(self, percentage, p0, p1):
        max_v, min_v = self._get_max_min()
        total = self.w * self.h
        pixels = int((total * percentage) / 100)
        candidates = random.sample(range(total), pixels)
        for each in candidates:
            ran = random.random()
            x = each // self.h
            y = each % self.h
            if ran < p0:
                self.img[x][y] = min_v
            elif ran > p1:
                self.img[x][y] = max_v

    def _get_sorrounding(self, x, y, size):
        half = size // 2
        r = range(-(half), half + 1)
        m = [[0 for i in range(size)] for j in range(size)]
        for delta_y in list(r):
            for delta_x in list(r):
                i = x + delta_x
                j = y + delta_y
                i = 0 if i < 0 else (self.w - 1 if i > self.w - 1 else i)
                j = 0 if j < 0 else (self.h - 1 if j > self.h - 1 else j)
                m[delta_x + half][delta_y + half] = self.img[i][j]
        return m

    def _common_filter(self, size, f):
        aux_matrix = [[0 for i in range(self.h)] for j in range(self.w)]
        for i in range(self.w):
            for j in range(self.h):
                m = self._get_sorrounding(i, j, size)
                aux_matrix[i][j] = f(m)
        return aux_matrix

    def mean_filter(self, size):
        def f(m):
            l = flat_matrix(m)
            return sum(l) / len(l)
        self.img = self._common_filter(size, f)

    def median_filter(self, size):
        def f(m):
            l = flat_matrix(m)
            return median(l)
        self.img = self._common_filter(size, f)

    def gauss_filter(self, size, sigma):
        def f(m):
            aux = 0
            for i in range(size):
                for j in range(size):
                    coe = (1 / (2 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, - \
                           (math.pow(i, 2) + math.pow(j, 2)) / math.pow(sigma, 2))
                    aux += coe * m[i][j]
            return aux
        self.img = self._common_filter(size, f)

    def border_filter(self, size):
        def f(m):
            coe = 1 / (size**2)
            half = size // 2
            aux = coe * m[half][half] * ((size**2) - 1)
            for i in range(size):
                for j in range(size):
                    if not i == half or not j == half:
                        aux -= coe * m[i][j]
            return aux
        self.img = self._common_filter(size, f)

    def _get_prewitt_matrix_x(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[j][0]
            for j in range(3):
                aux+= m[j][2]
            return aux
        return self._common_filter(3, f)

    def _get_prewitt_matrix_y(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[0][j]
            for j in range(3):
                aux+= m[2][j]
            return aux
        return self._common_filter(3, f)

    def _get_sobel_matrix_x(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[j][0] * ( 2 if j == 1 else 1 )
            for j in range(3):
                aux += m[j][2] * ( 2 if j == 1 else 1 )
            return aux
        return self._common_filter(3, f)

    def _get_sobel_matrix_y(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[0][j] * ( 2 if j == 1 else 1 )
            for j in range(3):
                aux += m[2][j] * ( 2 if j == 1 else 1 )
            return aux
        return self._common_filter(3, f)

    def _common_border_method(self, matrix_x, matrix_y):
        for i in range(self.w):
            for j in range(self.h):
                self.img[i][j] = math.sqrt(matrix_x[i][j]**2 + matrix_y[i][j]**2)

    def prewitt_method(self):
        self._common_border_method(self._get_prewitt_matrix_x(), self._get_prewitt_matrix_y())

    def sobel_method(self):
        self._common_border_method(self._get_sobel_matrix_x(), self._get_sobel_matrix_y())


class RGBImageAbstraction(ImageAbstraction):

    def get_image_list(self):
        flat_list = []
        for j in range(self.h):
            for i in range(self.w):
                flat_list.extend(list(self.img[i][j]))
        return flat_list

    def update_pixel(self, x, y, color):
        raise Exception("Not implemented on RGB")

    def get_pixel_color(self, x, y):
        c = self.img[x][y]
        return c

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
        return 'RGB'

    def get_bw_img(self):
        f = lambda x: int(sum(x)/3)
        aux_img = map_matrix(self.img, self.w, self.h, f)
        return BWImageAbstraction(flat_img_matrix(aux_img, self.w, self.h), self.get_size_tuple())

    def is_bw(self):
        return False
