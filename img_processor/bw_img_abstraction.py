import math
import random
from . import utils
from collections import Counter
from statistics import median
from .img_abstraction import ImageAbstraction

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
                    int(utils.transform_to_std(min_v, max_v, self.img[i][j])))
        return flat_list

    def update_pixel(self, x, y, color):
        max_v, min_v = self._get_max_min()
        v = utils.transform_from_std(min_v, max_v, color)
        self.img[x][y] = v

    def get_pixel_color(self, x, y):
        max_v, min_v = self._get_max_min()
        c = int(utils.transform_to_std(min_v, max_v, self.img[x][y]))
        return (c, c, c)

    def get_bw_img(self):
        return self

    def add(self, image):
        if not self.get_size_tuple() == image.get_size_tuple():
            raise Exception("Not same size")
        self.img = utils.add_matrix(self.img, image.img, self.w, self.h)

    def substract(self, image):
        if not self.get_size_tuple() == image.get_size_tuple():
            raise Exception("Not same size")
        self.img = utils.substract_matrix(self.img, image.img, self.w, self.h)

    def multiply(self, image):
        if not self.h == image.w:
            raise Exception("Not valid for product")
        self.img = utils.multiply_matrix(self.img, image.img, self.w, self.h,
                                   image.h)

    def negative(self):
        max_v, min_v = self._get_max_min()
        f = lambda x: max_v - x + min_v
        self.img = utils.map_matrix(self.img, self.w, self.h, f)

    def umbral(self, value):
        max_v, min_v = self._get_max_min()
        v = utils.transform_from_std(min_v, max_v, value)
        self.img = utils.map_matrix(self.img, self.w, self.h,
                              lambda x: max_v if x > v else min_v)

    def get_global_umbral(self):
        normalized_img_list = self.get_image_list()
        u = 128
        it = 0
        record = []
        while True:
            count = [0, 0]
            sum_p = [0, 0]
            for p in normalized_img_list:
                i = 0 if p < u else 1
                count[i] += 1
                sum_p[i] += p
            m1 = sum_p[0] / count[0]
            m2 = sum_p[1] / count[1]
            new_u = int((m1 + m2) / 2)
            it += 1
            if abs(new_u - u) <= 1:
                return (new_u, it)
            elif new_u in record:
                return (sum(record) // len(record), it)
            else:
                u = new_u
                record.append(new_u)

    def get_otsu_umbral(self):
        normalized_img_list = self.get_image_list()
        total = len(normalized_img_list)
        bucket_p = [0] * 256
        for each in normalized_img_list:
            bucket_p[each] += 1 / total
        bucket_p1 = [0] * 256
        bucket_m = [0] * 256
        for i, v in enumerate(bucket_p):
            bucket_p1[i] = v + (bucket_p1[i - 1] if i > 0 else 0)
            bucket_m[i] = i * v + (bucket_m[i - 1] if i > 0 else 0)
        m_g = bucket_m[-1]
        max_variance = -1
        index = -1
        for i in range(256):
            if bucket_p1[i] > 0 and bucket_p1[i] < 1:
                curr_variance = ((m_g * bucket_p1[i] - bucket_m[i])**2) / (
                    bucket_p1[i] * (1 - bucket_p1[i]))
                if curr_variance > max_variance:
                    max_variance, index = curr_variance, i
        return index

    def enhance_contrast(self, r1, r2):
        max_v, min_v = self._get_max_min()
        v1 = utils.transform_from_std(min_v, max_v, r1)
        v2 = utils.transform_from_std(min_v, max_v, r2)
        self.img = utils.map_matrix(
            self.img, self.w, self.h,
            lambda x: 0.5 * x if x < v1 else 2 * x if x > v2 else x)

    def power(self, value):
        max_v, min_v = self._get_max_min()
        self.img = utils.map_matrix(
            self.img, self.w, self.h,
            lambda x: (255 / pow(max_v, value)) * pow(x, value))

    def product(self, value):
        self.img = utils.map_matrix(self.img, self.w, self.h, lambda x: x * value)

    def compress(self):
        max_v, min_v = self._get_max_min()
        self.img = utils.map_matrix(
            self.img, self.w, self.h,
            lambda x: ((255) / (math.log(256))) * math.log(1 + utils.transform_to_std(min_v, max_v, x)))

    def equalize(self):
        normalized_img_list = self.get_image_list()
        equalized_list = ImageAbstraction.equalize_band(normalized_img_list)
        self.img = ImageAbstraction._get_img_matrix(self.w, self.h,
                                                    equalized_list)

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
        aux_matrix = ImageAbstraction._get_img_matrix(self.w, self.h,
                                                      normalized_img)
        total = self.w * self.h
        pixels = int((total * percentage) / 100)
        noise_list = []
        for i in range((pixels + 1) // 2):
            x1, x2 = random.random(), random.random()
            noise_list.append(
                math.sqrt(-2 * math.log(x1)) * math.cos(2 * math.pi * x2))
            noise_list.append(
                math.sqrt(-2 * math.log(x1)) * math.sin(2 * math.pi * x2))
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

    def mean_filter(self, size):
        def f(m):
            l = utils.flat_matrix(m)
            return sum(l) / len(l)

        self.img = self._common_filter(size, f)

    def median_filter(self, size):
        def f(m):
            l = utils.flat_matrix(m)
            return median(l)

        self.img = self._common_filter(size, f)

    def _apply_gauss_filter(self, img, size, sigma):
        def f(m):
            aux = 0
            half = size // 2
            r = range(-(half), (half) + 1)
            for i in r:
                for j in r:
                    coe = (1 / (2 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, - \
                           (math.pow(i, 2) + math.pow(j, 2)) / math.pow(sigma, 2))
                    aux += coe * m[i + half][j + half]
            return aux

        return self._common_filter_to_img(img, size, f)

    def gauss_filter(self, size, sigma):
        self.img = self._apply_gauss_filter(self.img, size, sigma)

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

    def _get_4_neighbors(img, i, j, w, h):
        ret = []
        ret.append(img[i - 1][j] if i > 0 else img[0][j])
        ret.append(img[i + 1][j] if i < w - 1 else img[w - 1][j])
        ret.append(img[i][j - 1] if j > 0 else img[i][0])
        ret.append(img[i][j + 1] if j < h - 1 else img[i][h - 1])
        return ret

    def _common_anisotropic_diffusion(self, f, times):
        normalized_img = self.get_image_list()
        aux_matrix = ImageAbstraction._get_img_matrix(self.w, self.h,
                                                      normalized_img)
        empty_matrix = [[0 for i in range(self.h)] for j in range(self.w)]
        for t in range(times):
            for i in range(self.w):
                for j in range(self.h):
                    curr_sum = 0
                    for neigh in BWImageAbstraction._get_4_neighbors(
                            aux_matrix, i, j, self.w, self.h):
                        d = neigh - aux_matrix[i][j]
                        curr_sum += d * f(d)
                    empty_matrix[i][j] = aux_matrix[i][j] + 0.25 * curr_sum
            aux_matrix = empty_matrix
        return aux_matrix

    def leclerc_anisotropic_diffusion(self, sigma, times):
        self.img = self._common_anisotropic_diffusion(
            lambda x: 1 / ((x**2 / sigma**2) + 1), times)

    def lorentziano_anisotropic_diffusion(self, sigma, times):
        self.img = self._common_anisotropic_diffusion(
            lambda x: math.e**(-(x**2) / sigma**2), times)

    def isotropic_diffusion(self, times):
        self.img = self._common_anisotropic_diffusion(lambda x: 1, times)

    def _get_prewitt_matrix_x(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[j][0]
            for j in range(3):
                aux += m[j][2]
            return aux

        return self._common_filter(3, f)

    def _get_prewitt_matrix_y(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[0][j]
            for j in range(3):
                aux += m[2][j]
            return aux

        return self._common_filter(3, f)

    def _get_sobel_matrix_x(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[j][0] * (2 if j == 1 else 1)
            for j in range(3):
                aux += m[j][2] * (2 if j == 1 else 1)
            return aux

        return self._common_filter(3, f)

    def _get_sobel_matrix_y(self):
        def f(m):
            aux = 0
            for j in range(3):
                aux -= m[0][j] * (2 if j == 1 else 1)
            for j in range(3):
                aux += m[2][j] * (2 if j == 1 else 1)
            return aux

        return self._common_filter(3, f)

    def _common_border_method(self, matrix_x, matrix_y):
        for i in range(self.w):
            for j in range(self.h):
                self.img[i][j] = math.sqrt(matrix_x[i][j]**2 + matrix_y[i][j]**
                                           2)

    def _common_directional_method(self, size, matrix_list):
        def f(m):
            candidates = [utils.put_mask(m, each, size) for each in matrix_list]
            max_with_index = max(
                [(abs(v), i) for i, v in enumerate(candidates)],
                key=lambda x: x[0])
            return candidates[max_with_index[1]]

        return self._common_filter(size, f)

    def kirsh_directional_method(self):
        self.img = self._common_directional_method(
            3, utils.get_kirsh_directional_matrix())

    def prewitt_directional_method(self):
        self.img = self._common_directional_method(
            3, utils.get_prewitt_directional_matrix())

    def sobel_directional_method(self):
        self.img = self._common_directional_method(
            3, utils.get_sobel_directional_matrix())

    def alternative_directional_method(self):
        self.img = self._common_directional_method(
            3, utils.get_alternative_directional_matrix())

    def canny_method(self):
        img_x = self._get_sobel_matrix_x()
        img_y = self._get_sobel_matrix_y()
        img_g = [[0 for i in range(self.h)] for j in range(self.w)]
        angles = [[0 for i in range(self.h)] for j in range(self.w)]

        for i in range(self.w):
            for j in range(self.h):
                img_g[i][j] = math.sqrt(img_x[i][j]**2 + img_y[i][j]**2)
                angles[i][j] = utils.get_angle(math.degrees(math.atan2(img_x[i][j], img_y[i][j])))

        for i in range(self.w):
            for j in range(self.h):
                self.img[i][j] = img_g[i][j]
                if img_g[i][j] > 0:
                    surr = self._get_sorrounding(img_g, i, j, 3)
                    alligned_pixels = utils.get_alligned_pixels(surr, angles[i][j])
                    if alligned_pixels[0] > img_g[i][j] or alligned_pixels[1] > img_g[i][j]:
                        self.img[i][j] = 0

    def canny_hysteresis_method(self, t1, t2):
        self.canny_method()
        
        aux = [[0 for i in range(self.h)] for j in range(self.w)]
        for i in range(self.w):
            for j in range(self.h):
                aux[i][j] = self.img[i][j]

        for i in range(self.w):
            for j in range(self.h):
                if self.img[i][j] <= t1:
                    aux[i][j] = 0

                elif self.img[i][j] <= t2:
                    surr = self._get_sorrounding(self.img, i, j, 3)
                    if not (surr[1][0] > 0 or surr[0][1] > 0 or surr[2][1] > 0 or surr[1][2] > 0):
                        aux[i][j] = 0
        self.img = aux

    def _get_laplacian_img_mask(self):
        def f(m):
            mask = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
            return utils.put_mask(m, mask, 3)

        return self._common_filter(3, f)

    def _get_LoG_img_mask(self, size, sigma):
        def f(m):
            aux = 0
            half = size // 2
            r = range(-(half), (half) + 1)
            for i in r:
                for j in r:
                    coe = -(1 / (math.pi * sigma**4)) * (1 - (
                        (i**2 + j**2) /
                        (2 * (sigma**2)))) * (math.pow(math.e, -(i**2 + j**2) /
                                                       (2 * (sigma**2))))
                    aux += coe * m[i + half][j + half]
            return aux

        return self._common_filter(size, f)

    def LoG_mask(self, size, sigma):
        self.img = self._get_LoG_img_mask(size, sigma)

    def laplacian_mask(self):
        self.img = self._get_laplacian_img_mask()

    def laplacian_method(self):
        aux = self._get_laplacian_img_mask()
        f = lambda curr, right, down: utils.sign(curr) != utils.sign(right) or utils.sign(curr) != utils.sign(down)
        self._mark_borders(aux, f)

    def _mark_borders(self, aux, condition):
        for i in range(self.w):
            for j in range(self.h):
                right_pixel = aux[i + 1][j] if i < self.w - 1 else aux[i][j]
                down_pixel = aux[i][j + 1] if j < self.h - 1 else aux[i][j]
                curr_pixel = aux[i][j]
                self.img[i][j] = 255 if condition(curr_pixel, right_pixel,
                                                  down_pixel) else 0

    def _mark_borders_with_umbral(self, aux_img, umbral):
        f = lambda curr, right, down: (utils.sign(curr) != utils.sign(right) and abs(curr) + abs(right) > umbral) or (utils.sign(curr) != utils.sign(down) and abs(curr) + abs(down) > umbral)
        self._mark_borders(aux_img, f)

    def laplacian_pending_method(self, umbral):
        aux = self._get_laplacian_img_mask()
        # max_v, min_v = (max_matrix(aux), min_matrix(aux))
        # u = utils.transform_from_std(min_v, max_v, umbral)
        self._mark_borders_with_umbral(aux, umbral)

    def LoG_method(self, size, sigma, umbral):
        aux = self._get_LoG_img_mask(size, sigma)
        # max_v, min_v = (max_matrix(aux), min_matrix(aux))
        # u = utils.transform_from_std(min_v, max_v, umbral)
        self._mark_borders_with_umbral(aux, umbral)

    def harris_method(self, umbral):
        pixel_list = []
        
        I_x = self._get_sobel_matrix_x()
        I_y = self._get_sobel_matrix_y()
        I_xx = utils.map_matrix(I_x, self.w, self.h, lambda x: x**2)
        I_yy = utils.map_matrix(I_y, self.w, self.h, lambda x: x**2)
        I_xy = utils.in_place_multiplication(I_x, I_y, self.w, self.h)

        I_xx = self._apply_gauss_filter(I_xx, 7, 2)
        I_yy = self._apply_gauss_filter(I_yy, 7, 2)
        I_xy = self._apply_gauss_filter(I_xy, 7, 2)

        empty_matrix = [[0 for i in range(self.h)] for j in range(self.w)]
        k = 0.04

        for i in range(self.w):
            for j in range(self.h):
                empty_matrix[i][j] = (I_xx[i][j]*I_yy[i][j] - (I_xy[i][j]**2)) - k*((I_xx[i][j]+I_yy[i][j])**2)

        max_energy = utils.max_matrix(empty_matrix)
        min_energy = utils.min_matrix(empty_matrix)

        diff = max_energy-min_energy

        for i in range(self.w):
            for j in range(self.h):
                if empty_matrix[i][j] > umbral*diff+min_energy:
                    pixel_list.append([i,j])

        return pixel_list

    def susan_method(self, umbral, reference):
        def f(m):
            aux = 0
            for j in range(3):
                aux += 1 if abs(m[j+2][0]-m[3][3])<umbral else 0
                aux += 1 if abs(m[j+2][6]-m[3][3])<umbral else 0
            for j in range(5):
                aux += 1 if abs(m[j+1][1]-m[3][3])<umbral else 0
                aux += 1 if abs(m[j+1][5]-m[3][3])<umbral else 0
            for j in range(7):
                aux += 1 if abs(m[j][2]-m[3][3])<umbral else 0
                aux += 1 if abs(m[j][3]-m[3][3])<umbral else 0
                aux += 1 if abs(m[j][4]-m[3][3])<umbral else 0
            return aux

        img_aux = self._common_filter(7, f)
        
        pixel_list = []
        for i in range(self.w):
            for j in range(self.h):
                if (1-img_aux[i][j]/37) > reference:
                    pixel_list.append([i,j])

        return pixel_list

    def hugh_for_lines(self, o, p, epsilon):
        ret = []
        votes = {}
        d = max(self.w, self.h)
        for i in utils.drange(p[0], p[2], p[1]):
            votes[i] = {}
            for j in utils.drange(o[0], o[2], o[1]):
                votes[i][j] = 0

        max_v, min_v = self._get_max_min()
        for i in range(self.w):
            for j in range(self.h):
                x = i
                y = self.h - 1 - j
                if self.img[x][y] == max_v:
                    for p in votes.keys():
                        for o in votes[p].keys():
                            if abs(x*math.cos(math.radians(o)) + y*math.sin(math.radians(o)) - p) < epsilon:
                                votes[p][o] = votes[p][o]+1
        max_votes = -1
        best_line = None
        for p in votes.keys():
            for o in votes[p].keys():
                if votes[p][o] > max_votes:
                    max_votes = votes[p][o]
                    best_line = (p,o)

        for p in votes.keys():
            for o in votes[p].keys():
                if votes[p][o] > 0.8*max_votes:
                    ret.append((p,o))

        return ret

    def hugh_for_circles(self, p, r, epsilon):
        ret = []
        votes = {}
        for a in utils.drange(p[0], p[2], p[1]):
            votes[a] = {}
            for b in utils.drange(p[0], p[2], p[1]):
                votes[a][b] = {}
                for rad in utils.drange(r[0], r[2], r[1]):
                    votes[a][b][rad] = 0

        max_v, min_v = self._get_max_min()

        for i in range(self.w):
            for j in range(self.h):
                x = i
                y = self.h - 1 - j
                if self.img[x][y] == max_v:
                    for a in votes.keys():
                        for b in votes[a].keys():
                            for r in votes[a][b].keys():
                                if abs((x-a)**2 + (y-b)**2 - r**2) < epsilon:
                                    votes[a][b][r] = votes[a][b][r]+1
        max_votes = -1
        best_line = None
        for a in votes.keys():
            for b in votes[a].keys():
                for r in votes[a][b].keys():
                    if votes[a][b][r] > max_votes:
                        max_votes = votes[a][b][r]
                        best_line = (a,b,r)

        for a in votes.keys():
            for b in votes[a].keys():
                for r in votes[a][b].keys():
                    if votes[a][b][r] > 0.8*max_votes:
                        ret.append((a,b,r))

        return ret

    def get_mean(self, phi):
        ret = 0
        count = 0
        for i in range(self.w):
            for j in range(self.h):
                if phi[i][j] == -3:
                    ret += self.img[i][j]
                    count += 1

        return ret/count if count > 0 else 0

    def get_f(self, pixel, mean, probability):
        p = 1 - (abs(self.img[pixel[0]][pixel[1]] - mean)/(255))
        return -1 if p < probability else 1

