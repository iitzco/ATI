import math
import random
import copy
from . import utils
from collections import Counter
from statistics import median

gauss5x5 = [
                [1/273,4/273,7/273,4/273,1/273],
                [4/273,16/273,26/273,16/273,4/273],
                [7/273,26/273,41/273,26/273,7/273],
                [4/273,16/273,26/273,16/273,4/273],
                [1/273,4/273,7/273,4/273,1/273]
            ]

gauss3x3 = [
                [0.0585, 0.0965, 0.0585],
                [0.0965, 0.1591, 0.0965],
                [0.0585, 0.0965, 0.0585]
            ]

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
        return (utils.max_matrix(self.img), utils.min_matrix(self.img))

    def get_image_list(self):
        pass

    def get_image_bytes(self):
        return bytes(self.get_image_list())

    def get_size_tuple(self):
        return (self.w, self.h)

    def _get_sorrounding(self, img, x, y, size):
        half = size // 2
        r = range(-(half), half + 1)
        m = [[0 for i in range(size)] for j in range(size)]
        for delta_y in list(r):
            for delta_x in list(r):
                i = x + delta_x
                j = y + delta_y
                i = 0 if i < 0 else (self.w - 1 if i > self.w - 1 else i)
                j = 0 if j < 0 else (self.h - 1 if j > self.h - 1 else j)
                m[delta_x + half][delta_y + half] = img[i][j]
        return m

    def _common_filter_to_img(self, img, size, f):
        aux_matrix = [[0 for i in range(self.h)] for j in range(self.w)]
        for i in range(self.w):
            for j in range(self.h):
                m = self._get_sorrounding(img, i, j, size)
                aux_matrix[i][j] = f(m)
        return aux_matrix

    def _common_filter(self, size, f):
        return self._common_filter_to_img(self.img, size, f)

    def equalize_band(normalized_img_list):
        c = Counter(normalized_img_list)
        total = len(normalized_img_list)
        s_list = [0] * 256
        s_list[255] = total
        for i in range(1, 256):
            s_list[255 - i] = s_list[256 - i] - c[256 - i]
        s_list = [each / total for each in s_list]
        min_value = min(s_list)
        return list(
            map(lambda x: (int(((s_list[x] - min_value) / (1 - min_value)) * 255 + 0.5)),
                normalized_img_list))

    def prewitt_method(self):
        self._common_border_method(self._get_prewitt_matrix_x(),
                                   self._get_prewitt_matrix_y())

    def sobel_method(self):
        self._common_border_method(self._get_sobel_matrix_x(),
                                   self._get_sobel_matrix_y())

    def sobel_x_to_img(self):
        self.img = self._get_sobel_matrix_x()

    def sobel_y_to_img(self):
        self.img = self._get_sobel_matrix_y()

    def init_phi_matrix(self, lin, lout):
        ret = [[0 for i in range(self.h)] for j in range(self.w)]

        for each in lin:
            ret[each[0]][each[1]] = -1

        for each in lout:
            ret[each[0]][each[1]] = 1

        inner_pixel = self.find_empty(ret, lin, lout)
        if inner_pixel:
            self.fill_pixels(ret, inner_pixel, -3)

        # Outer pixels remain only.
        for i in range(self.w):
            for j in range(self.h):
                if ret[i][j] == 0:
                    ret[i][j] = 3

        return ret

    def find_empty(self, phi, border, other):
        for each in border:
            x, y = each
            if x-1 >=0 and phi[x-1][y] == 0:
                return (x-1, y)
            if x+1 < self.w and phi[x+1][y] == 0:
                return (x+1, y)
            if y-1 >=0 and phi[x][y-1] == 0:
                return (x, y-1)
            if y+1 < self.h and phi[x][y+1] == 0:
                return (x, y+1)

    def fill_pixels(self, phi, origin, elem):
        x, y = origin
        queue = [(x,y)]

        while queue:
            x,y = queue.pop()
            phi[x][y] = elem

            if x-1 >= 0 and phi[x-1][y] == 0:
                queue.append((x-1,y))
            if x+1 < self.w and phi[x+1][y] == 0:
                queue.append((x+1,y))
            if y-1 >= 0 and phi[x][y-1] == 0:
                queue.append((x,y-1))
            if y+1 < self.h and phi[x][y+1] == 0:
                queue.append((x,y+1))

    def get_fs(self, pixel, phi):
        m = self._get_sorrounding(phi, pixel[0], pixel[1], 3)
        mult = [[a*b for a,b in zip(m[i],gauss3x3[i])] for i in range(3)]
        return sum(map(sum, mult))

    # def get_fs(self, pixel, phi):
    #     m = self._get_sorrounding(phi, pixel[0], pixel[1], 5)
    #     mult = [[a*b for a,b in zip(m[i],gauss5x5[i])] for i in range(5)]
    #     return sum(map(sum, mult))

    def is_lin(self, pixel, phi):
        x, y = pixel

        if x-1 >= 0 and phi[x-1][y] > 0:
            return True
        if x+1 < self.w and phi[x+1][y] > 0:
            return True
        if y-1 >= 0 and phi[x][y-1] > 0:
            return True
        if y+1 < self.h and phi[x][y+1] > 0:
            return True
        return False

    def is_lout(self, pixel, phi):
        x, y = pixel

        if x-1 >= 0 and phi[x-1][y] < 0:
            return True
        if x+1 < self.w and phi[x+1][y] < 0:
            return True
        if y-1 >= 0 and phi[x][y-1] < 0:
            return True
        if y+1 < self.h and phi[x][y+1] < 0:
            return True
        return False

    def filter_lin(self, lin, phi):
        for each in list(lin):
            if not self.is_lin(each, phi):
                lin.remove(each)
                x, y = each
                phi[x][y] = -3

    def filter_lout(self, lout, phi):
        for each in list(lout):
            if not self.is_lout(each, phi):
                lout.remove(each)
                x, y = each
                phi[x][y] = 3

    def expand_contour(self, pixel, lin, lout, phi):
        lout.remove(pixel)
        lin.add(pixel)
        x,y = pixel
        phi[x][y] = -1

        if x-1 >= 0 and phi[x-1][y] == 3:
            lout.add((x-1, y))
            phi[x-1][y] = 1
        if x+1 < self.w and phi[x+1][y] == 3:
            lout.add((x+1, y))
            phi[x+1][y] = 1
        if y-1 >= 0 and phi[x][y-1] == 3:
            lout.add((x, y-1))
            phi[x][y-1] = 1
        if y+1 < self.h and phi[x][y+1] == 3:
            lout.add((x, y+1))
            phi[x][y+1] = 1

    def contract_contour(self, pixel, lin, lout, phi):
        lin.remove(pixel)
        lout.add(pixel)
        x, y = pixel
        phi[x][y] = 1

        if x-1 >= 0 and phi[x-1][y] == -3:
            lin.add((x-1, y))
            phi[x-1][y] = -1
        if x+1 < self.w and phi[x+1][y] == -3:
            lin.add((x+1, y))
            phi[x+1][y] = -1
        if y-1 >= 0 and phi[x][y-1] == -3:
            lin.add((x, y-1))
            phi[x][y-1] = -1
        if y+1 < self.h and phi[x][y+1] == -3:
            lin.add((x, y+1))
            phi[x][y+1] = -1

    def first_cycle(self, lin, lout, phi, mean, probability):
        for each in list(lout):
            f = self.get_f(each, mean, probability)
            if f > 0:
                self.expand_contour(each, lin, lout, phi)
        self.filter_lin(lin, phi)

        for each in list(lin):
            f = self.get_f(each, mean, probability)
            if f < 0:
                self.contract_contour(each, lin, lout, phi)
        self.filter_lout(lout, phi)

    def second_cycle(self, lin, lout, phi):
        for each in list(lout):
            f = self.get_fs(each, phi)
            if f < 0:
                self.expand_contour(each, lin, lout, phi)
        self.filter_lin(lin, phi)

        for each in list(lin):
            f = self.get_fs(each, phi)
            if f > 0:
                self.contract_contour(each, lin, lout, phi)
        self.filter_lout(lout, phi)

    def check_end(self, lin, lout, mean, probability):
        for each in lin:
            f = self.get_f(each, mean, probability)
            if f < 0:
                return False
        for each in lout:
            f = self.get_f(each, mean, probability)
            if f > 0:
                return False
        return True

    def contour_detection_method(self, lin, lout, nmax, phi, probability, full_tracking):
        if not phi:
            phi = self.init_phi_matrix(lin, lout)
        
        iterations = 0
        mean = self.get_mean(phi)

        while iterations < nmax:

            self.first_cycle(lin, lout, phi, mean, probability)

            if self.check_end(lin, lout, mean, probability):
                break

            if full_tracking:

                self.second_cycle(lin, lout, phi)

                if self.check_end(lin, lout, mean, probability):
                    break

            iterations+=1
        
        return lin, lout, phi
