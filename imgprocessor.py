import math
import random
from collections import Counter
from statistics import median


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
    return 1 if x>=0 else -1


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


def put_mask(original, mask, size):
    ret = 0
    for i in range(size):
        for j in range(size):
            ret += original[i][j]*mask[i][j]
    return ret


def get_kirsh_directional_matrix():
    # 5  5  5    5  5 -3  -3 -3 -3   5 -3 -3   
    # -3 0 -3    5  0 -3   5  0 -3   5  0 -3   
    # -3 -3 -3  -3 -3 -3   5  5 -3   5 -3 -3  
    ret = []
    ret.append([[5, -3, -3], [5, 0, -3], [5, -3, -3]]) 
    ret.append([[5,5,-3], [5,0,-3], [-3, -3, -3]])
    ret.append([[-3,5,5], [-3, 0, 5], [-3,-3,-3]])
    ret.append([[5,5,5], [-3, 0, -3], [-3, -3, -3]])
    return ret

def get_prewitt_directional_matrix():
    # 1  1  1    1  1  0    0 -1 -1    1  0 -1
    # 0  0  0    1  0 -1    1  0  1    1  0 -1
    # -1 -1 -1   0 -1 -1    1  1  0    1  0 -1
    ret = []
    ret.append([[1,0,-1], [1, 0, -1], [1,0,-1]])
    ret.append([[1,1,0],[1,0,-1],[0,-1,-1]])
    ret.append([[0,1,1], [-1,0,1], [-1,-1,0]])
    ret.append([[1,1,1], [0,0,0], [-1,-1,-1]])
    return ret

def get_sobel_directional_matrix():
    # 1  2  1    2  1  0    0 -1 -2    1  0 -1
    # 0  0  0    1  0 -1    1  0  1    2  0 -2
    # -1 -2 -1   0 -1 -2    2  1  0    1  0 -1
    ret = []
    ret.append([[1,0,-1], [2, 0, -2], [1,0,-1]])
    ret.append([[2,1,0],[1,0,-1],[0,-1,-2]])
    ret.append([[0,1,2], [-1,0,1], [-2,-1,0]])
    ret.append([[1,2,1], [0,0,0], [-1,-2,-1]])
    return ret

def get_alternative_directional_matrix():
    # 1  1  1    1  1  1    1 -1 -1   1  1 -1    
    # 1 -2  1    1 -2 -1    1 -2 -1   1 -2 -1    
    # -1 -1 -1   1 -1 -1    1  1  1   1  1 -1    
    ret = []
    ret.append([[1,1, -1],[1,-2,-1],[1,-1,-1]])
    ret.append([[1,1,1],[1,-2,-1],[1,-1,-1]])
    ret.append([[1,1,1],[-1,-2,1],[-1,-1,1]])
    ret.append([[1,1,1], [1,-2,1],[-1,-1,-1]])
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

    def equalize_band(normalized_img_list):
        c = Counter(normalized_img_list)
        total = len(normalized_img_list)
        s_list = [0] * 256
        s_list[255] = total
        for i in range(1, 256):
            s_list[255 - i] = s_list[256 - i] - c[256 - i]
        s_list = [each / total for each in s_list]
        min_value = min(s_list)
        return list(map(lambda x: (int(
            ((s_list[x] - min_value) / (1 - min_value)) * 255 + 0.5)), normalized_img_list))

    def prewitt_method(self):
        self._common_border_method(
            self._get_prewitt_matrix_x(),
            self._get_prewitt_matrix_y())

    def sobel_method(self):
        self._common_border_method(
            self._get_sobel_matrix_x(),
            self._get_sobel_matrix_y())

    def sobel_x_to_img(self):
        self.img = self._get_sobel_matrix_x()

    def sobel_y_to_img(self):
        self.img = self._get_sobel_matrix_y()


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

    def get_global_umbral(self):
        normalized_img_list = self.get_image_list()
        u = 128
        it = 0
        record = []
        while True:
            count = [0,0]
            sum_p = [0,0]
            for p in normalized_img_list:
                i = 0 if p<u else 1
                count[i]+=1
                sum_p[i]+=p
            m1 = sum_p[0]/count[0]
            m2 = sum_p[1]/count[1]
            new_u = int((m1 + m2) / 2)
            it+=1
            if abs(new_u-u)<=1:
                return (new_u, it)
            elif new_u in record:
                return (sum(record)//len(record), it)
            else:
                u = new_u
                record.append(new_u)

    def get_otsu_umbral(self):
        normalized_img_list = self.get_image_list()
        total = len(normalized_img_list)
        bucket_p = [0]*256
        for each in normalized_img_list:
            bucket_p[each]+=1/total
        bucket_p1 = [0]*256
        bucket_m = [0]*256
        for i, v in enumerate(bucket_p):
            bucket_p1[i] = v + (bucket_p1[i-1] if i>0 else 0)
            bucket_m[i] = i*v + (bucket_m[i-1] if i>0 else 0)
        m_g = bucket_m[-1]
        max_variance = -1
        index = -1
        for i in range(256):
            if bucket_p1[i]>0 and bucket_p1[i]<1:
                curr_variance = ((m_g*bucket_p1[i]-bucket_m[i])**2)/(bucket_p1[i]*(1-bucket_p1[i]))
                if curr_variance > max_variance:
                    max_variance, index = curr_variance, i
        return index

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
        normalized_img_list = self.get_image_list()
        equalized_list = ImageAbstraction.equalize_band(normalized_img_list)
        self.img = ImageAbstraction._get_img_matrix(
            self.w, self.h, equalized_list)

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
            half = size//2
            r = range(-(half), (half)+1)
            for i in r:
                for j in r:
                    coe = (1 / (2 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, - \
                           (math.pow(i, 2) + math.pow(j, 2)) / math.pow(sigma, 2))
                    aux += coe * m[i+half][j+half]
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

    def _get_4_neighbors(img, i, j, w, h):
        ret = []
        ret.append(img[i-1][j] if i>0 else img[0][j])
        ret.append(img[i+1][j] if i<w-1 else img[w-1][j])
        ret.append(img[i][j-1] if j>0 else img[i][0])
        ret.append(img[i][j+1] if j<h-1 else img[i][h-1])
        return ret

    def _common_anisotropic_diffusion(self, f, times):
        normalized_img = self.get_image_list()
        aux_matrix = ImageAbstraction._get_img_matrix(
            self.w, self.h, normalized_img)
        empty_matrix = [[0 for i in range(self.h)] for j in range(self.w)]
        for t in range(times):
            for i in range(self.w):
                for j in range(self.h):
                    curr_sum=0
                    for neigh in BWImageAbstraction._get_4_neighbors(aux_matrix, i, j, self.w, self.h):
                        d = neigh - aux_matrix[i][j]
                        curr_sum += d*f(d)
                    empty_matrix[i][j] = aux_matrix[i][j] + 0.25*curr_sum
            aux_matrix = empty_matrix
        return aux_matrix

    def leclerc_anisotropic_diffusion(self, sigma, times):
        self.img = self._common_anisotropic_diffusion(lambda x : 1/((x**2/sigma**2)+1), times)

    def lorentziano_anisotropic_diffusion(self, sigma, times):
        self.img = self._common_anisotropic_diffusion(lambda x : math.e**(-(x**2)/sigma**2), times)

    def isotropic_diffusion(self, times):
        self.img = self._common_anisotropic_diffusion(lambda x : 1, times)

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
                self.img[i][j] = math.sqrt(
                    matrix_x[i][j]**2 + matrix_y[i][j]**2)

    def _common_directional_method(self, size, matrix_list):
        def f(m):
            candidates = [put_mask(m, each, size) for each in matrix_list]
            max_with_index = max([(abs(v),i) for i,v in enumerate(candidates)], key=lambda x:x[0])
            return candidates[max_with_index[1]]
        return self._common_filter(size, f)

    def kirsh_directional_method(self):
        self.img = self._common_directional_method(3, get_kirsh_directional_matrix())
        
    def prewitt_directional_method(self):
        self.img = self._common_directional_method(3, get_prewitt_directional_matrix())

    def sobel_directional_method(self):
        self.img = self._common_directional_method(3, get_sobel_directional_matrix())

    def alternative_directional_method(self):
        self.img = self._common_directional_method(3, get_alternative_directional_matrix())

    def _get_laplacian_img_mask(self):
        def f(m):
            mask = [[0,-1,0],[-1,4,-1],[0,-1,0]]
            return put_mask(m, mask, 3)
        return self._common_filter(3, f)

    def _get_LoG_img_mask(self, size, sigma):
        def f(m):
            aux = 0
            half = size//2
            r = range(-(half), (half)+1)
            for i in r:
                for j in r:
                    coe = -(1/(math.pi*sigma**4))*(1-((i**2+j**2)/(2*(sigma**2))))*(math.pow(math.e, -(i**2+j**2)/(2*(sigma**2))))
                    aux += coe * m[i+half][j+half]
            return aux
        return self._common_filter(size, f)

    def LoG_mask(self, size, sigma):
        self.img = self._get_LoG_img_mask(size, sigma)

    def laplacian_mask(self):
        self.img = self._get_laplacian_img_mask()

    def laplacian_method(self):
        aux = self._get_laplacian_img_mask()
        f = lambda curr, right, down: sign(curr)!=sign(right) or sign(curr)!=sign(down)
        self._mark_borders(aux, f)

    def _mark_borders(self, aux, condition):
        for i in range(self.w):
            for j in range(self.h):
                right_pixel = aux[i+1][j] if i<self.w-1 else aux[i][j]
                down_pixel = aux[i][j+1] if j<self.h-1 else aux[i][j]
                curr_pixel = aux[i][j]
                self.img[i][j] = 255 if condition(curr_pixel, right_pixel, down_pixel) else 0

    def _mark_borders_with_umbral(self, aux_img, umbral):
        f = lambda curr, right, down: (sign(curr)!=sign(right) and 
                    abs(curr)+abs(right)>umbral ) or (sign(curr)!=sign(down)
                            and abs(curr)+abs(down) > umbral)
        self._mark_borders(aux_img, f)

    def laplacian_pending_method(self, umbral):
        aux = self._get_laplacian_img_mask()
        # max_v, min_v = (max_matrix(aux), min_matrix(aux))
        # u = transform_from_std(min_v, max_v, umbral)
        self._mark_borders_with_umbral(aux, umbral)

    def LoG_method(self, size, sigma, umbral):
        aux = self._get_LoG_img_mask(size, sigma)
        # max_v, min_v = (max_matrix(aux), min_matrix(aux))
        # u = transform_from_std(min_v, max_v, umbral)
        self._mark_borders_with_umbral(aux, umbral)


class RGBImageAbstraction(ImageAbstraction):

    def get_image_list(self):
        flat_list = []
        max_min_bands = []
        for i in range(3):
            max_min_bands.append(self._get_max_min_in_band(i))
        for j in range(self.h):
            for i in range(self.w):
                for x in range(3):
                    flat_list.append(int(transform_to_std(max_min_bands[x][
                        1], max_min_bands[x][0], self.img[i][j][x])))
        return flat_list

    def _get_max_min_in_band(self, band):
        return (
            max_matrix_band(
                self.img, band), min_matrix_band(
                self.img, band))

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
        self.img = map_matrix(self.img, self.w, self.h, f)

    def get_mode(self):
        return 'RGB'

    def get_bw_img(self):
        f = lambda x: int(sum(x) / 3)
        aux_img = map_matrix(self.img, self.w, self.h, f)
        return BWImageAbstraction(
            flat_img_matrix(
                aux_img,
                self.w,
                self.h),
            self.get_size_tuple())

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
        eq_list_r, eq_list_g, eq_list_b = map(
            ImageAbstraction.equalize_band, bands)
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
                    aux[each] = math.sqrt(
                        matrix_x[i][j][each]**2 + matrix_y[i][j][each]**2)
                self.img[i][j] = tuple(map(int, aux))
