import math
import random
from . import utils
from collections import Counter
from statistics import median
from .img_abstraction import ImageAbstraction
from .bw_img_abstraction import BWImageAbstraction


class RGBImageAbstraction(ImageAbstraction):
    def get_image_list(self):
        flat_list = []
        max_min_bands = []
        for i in range(3):
            max_min_bands.append(self._get_max_min_in_band(i))
        for j in range(self.h):
            for i in range(self.w):
                for x in range(3):
                    flat_list.append(
                        int(
                            utils.transform_to_std(max_min_bands[x][
                                1], max_min_bands[x][0], self.img[i][j][x])))
        return flat_list

    def _get_max_min_in_band(self, band):
        return (utils.max_matrix_band(self.img, band), utils.min_matrix_band(self.img,
                                                                 band))

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
        max_min_bands = []
        for i in range(3):
            max_min_bands.append(self._get_max_min_in_band(i))

        def f(t):
            l = []
            for i, e in enumerate(t):
                l.append(max_min_bands[i % 3][0] - e + max_min_bands[i % 3][1])
            return tuple(l)

        self.img = utils.map_matrix(self.img, self.w, self.h, f)

    def get_mode(self):
        return 'RGB'

    def get_bw_img(self):
        f = lambda x: int(sum(x) / 3)
        aux_img = utils.map_matrix(self.img, self.w, self.h, f)
        return BWImageAbstraction(
            utils.flat_img_matrix(aux_img, self.w, self.h), self.get_size_tuple())

    def is_bw(self):
        return False

    def get_bands(self):
        img_list = self.get_image_list()
        img_r, img_g, img_b = [], [], []
        for i, e in enumerate(img_list):
            if (i % 3 == 0):
                img_r.append(e)
            elif (i % 3 == 1):
                img_g.append(e)
            else:
                img_b.append(e)
        return (img_r, img_g, img_b)

    def combine_bands_list(r, g, b):
        ret = []
        for i in range(len(r)):
            ret.append((r[i], g[i], b[i]))
        return ret

    def equalize(self):
        img_list = self.get_image_list()
        bands = self.get_bands()
        eq_list_r, eq_list_g, eq_list_b = map(ImageAbstraction.equalize_band,
                                              bands)
        self.img = ImageAbstraction._get_img_matrix(
            self.w, self.h, RGBImageAbstraction.combine_bands_list(
                eq_list_r, eq_list_g, eq_list_b))

    def _get_prewitt_matrix_x(self):
        def f(m):
            aux = (0, 0, 0)
            for j in range(3):
                aux = tuple(x - y for x, y in zip(aux, m[j][0]))
            for j in range(3):
                aux = tuple(x + y for x, y in zip(aux, m[j][2]))
            return aux

        return self._common_filter(3, f)

    def _get_prewitt_matrix_y(self):
        def f(m):
            aux = (0, 0, 0)
            for j in range(3):
                tuple(x - y for x, y in zip(aux, m[0][j]))
            for j in range(3):
                tuple(x + y for x, y in zip(aux, m[2][j]))
            return aux

        return self._common_filter(3, f)

    def _get_sobel_matrix_x(self):
        def f(m):
            aux = (0, 0, 0)
            for j in range(3):
                aux = tuple(x - y * (2 if j == 1 else 1)
                            for x, y in zip(aux, m[j][0]))
            for j in range(3):
                aux = tuple(x + y * (2 if j == 1 else 1)
                            for x, y in zip(aux, m[j][2]))
            return aux

        return self._common_filter(3, f)

    def _get_sobel_matrix_y(self):
        def f(m):
            aux = (0, 0, 0)
            for j in range(3):
                tuple(x - y * (2 if j == 1 else 1)
                      for x, y in zip(aux, m[0][j]))
            for j in range(3):
                tuple(x + y * (2 if j == 1 else 1)
                      for x, y in zip(aux, m[2][j]))
            return aux

        return self._common_filter(3, f)

    def _common_border_method(self, matrix_x, matrix_y):
        for i in range(self.w):
            for j in range(self.h):
                aux = [0] * 3
                for each in range(3):
                    aux[each] = math.sqrt(matrix_x[i][j][each]**2 + matrix_y[i]
                                          [j][each]**2)
                self.img[i][j] = tuple(map(int, aux))

    def get_mean(self, phi):
        ret = [0, 0, 0]
        count = 0
        for i in range(self.w):
            for j in range(self.h):
                if phi[i][j] == -3:
                    ret[0] += self.img[i][j][0]
                    ret[1] += self.img[i][j][1]
                    ret[2] += self.img[i][j][2]
                    count += 1

        return (ret[0]/count, ret[1]/count, ret[2]/count) if count > 0 else (0,0,0)

    def get_f(self, pixel, mean, probability):
        x, y = pixel
        norm = math.sqrt((self.img[x][y][0] - mean[0])**2 + (self.img[x][y][1] - mean[1])**2 +(self.img[x][y][2] - mean[2])**2)
        p = 1 - (norm/math.sqrt(3*(255**2)))
        return -1 if p < probability else 1
