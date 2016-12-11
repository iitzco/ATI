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

    def expand_contour(self, pixel, tracking_container):
        tracking_container.lout.remove(pixel)
        tracking_container.lin.add(pixel)
        x,y = pixel
        tracking_container.phi[x][y] = -1

        if x-1 >= 0 and tracking_container.phi[x-1][y] == 3:
            tracking_container.lout.add((x-1, y))
            tracking_container.phi[x-1][y] = 1
        if x+1 < self.w and tracking_container.phi[x+1][y] == 3:
            tracking_container.lout.add((x+1, y))
            tracking_container.phi[x+1][y] = 1
        if y-1 >= 0 and tracking_container.phi[x][y-1] == 3:
            tracking_container.lout.add((x, y-1))
            tracking_container.phi[x][y-1] = 1
        if y+1 < self.h and tracking_container.phi[x][y+1] == 3:
            tracking_container.lout.add((x, y+1))
            tracking_container.phi[x][y+1] = 1

    def contract_contour(self, pixel, tracking_container):
        tracking_container.lin.remove(pixel)
        tracking_container.lout.add(pixel)
        x, y = pixel
        tracking_container.phi[x][y] = 1

        if x-1 >= 0 and tracking_container.phi[x-1][y] == -3:
            tracking_container.lin.add((x-1, y))
            tracking_container.phi[x-1][y] = -1
        if x+1 < self.w and tracking_container.phi[x+1][y] == -3:
            tracking_container.lin.add((x+1, y))
            tracking_container.phi[x+1][y] = -1
        if y-1 >= 0 and tracking_container.phi[x][y-1] == -3:
            tracking_container.lin.add((x, y-1))
            tracking_container.phi[x][y-1] = -1
        if y+1 < self.h and tracking_container.phi[x][y+1] == -3:
            tracking_container.lin.add((x, y+1))
            tracking_container.phi[x][y+1] = -1

    def first_cycle(self, tracking_container):
        for each in list(tracking_container.lout):
            f = self.get_f(each, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking)
            if f > 0:
                self.expand_contour(each, tracking_container)
        self.filter_lin(tracking_container.lin, tracking_container.phi)

        for each in list(tracking_container.lin):
            f = self.get_f(each, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking)
            if f < 0:
                self.contract_contour(each, tracking_container)
        self.filter_lout(tracking_container.lout, tracking_container.phi)

    def second_cycle(self, tracking_container):
        for each in list(tracking_container.lout):
            f = self.get_fs(each, tracking_container.phi)
            if f < 0:
                self.expand_contour(each, tracking_container)
        self.filter_lin(tracking_container.lin, tracking_container.phi)

        for each in list(tracking_container.lin):
            f = self.get_fs(each, tracking_container.phi)
            if f > 0:
                self.contract_contour(each, tracking_container)
        self.filter_lout(tracking_container.lout, tracking_container.phi)

    def check_end(self, tracking_container):
        for each in tracking_container.lin:
            f = self.get_f(each, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking)
            if f < 0:
                return False
        for each in tracking_container.lout:
            f = self.get_f(each, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking)
            if f > 0:
                return False
        return True

    def contour_detection_method(self, tracking_container):
        iterations = 0

        while iterations < tracking_container.nmax:

            self.first_cycle(tracking_container)

            if self.check_end(tracking_container):
                break

            if tracking_container.full_tracking:

                self.second_cycle(tracking_container)

                if self.check_end(tracking_container):
                    break

            iterations+=1

    def analyze_possible_oclussion(self, tracking_container, displacement, center_mass):
        d = utils.vector_to_versor(displacement)
        diagonal = math.sqrt(self.w**2 + self.h**2)

        # Moves each 10%
        for percent in range(1,11):
            movement = d[0]*diagonal*(percent/10), d[1]*diagonal*(percent/10)
            new_center = int(center_mass[0] + movement[0]), int(center_mass[1] + movement[1])

            if not self.inside_image(new_center):
                return

            f = self.get_f(new_center, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking)
            if f > 0:
                if self.surrounding_in_contour(new_center, tracking_container.mean, tracking_container.probability, tracking_container.hsv_tracking):
                    tracking_container.lin, tracking_container.lout = self.initialize_box(new_center)
                    tracking_container.phi = self.init_phi_matrix(tracking_container.lin, tracking_container.lout)
                    tracking_container.reset()
                    self.contour_detection_method(tracking_container)
                    return
                                

    def surrounding_in_contour(self, center, mean, probability, hsv_tracking):
        # Analyze 10x10 surrounding box
        for i in range(center[0]-5, center[0]+6):
            for j in range(center[1]-5, center[1]+6):
                p = i, j
                if self.inside_image(p):
                    if not self.get_f(p, mean, probability, hsv_tracking):
                        return False
        return True

    def initialize_box(self, center):
        lout, lin = set(), set()
        for i in range(center[0]-1, center[0]+2):
            lin.add((i, center[1] - 1))
            lin.add((i, center[1] + 1))
            if i!=center[0]:
                lin.add((i, center[1]))
        for i in range(center[0]-2, center[0]+3):
            lout.add((i, center[1] - 2))
            lout.add((i, center[1] + 2))
            if abs(i-center[0]) == 2:
                lout.add((i, center[1]-1))
                lout.add((i, center[1]))
                lout.add((i, center[1]+1))

        return lin, lout

    def inside_image(self, p):
        return p[0] >= 0 and p[0] < self.w and p[1] >=0 and p[1] < self.h
