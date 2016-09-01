def max_matrix(matrix):
    return max([max(each) for each in matrix])

def min_matrix(matrix):
    return min([min(each) for each in matrix])

def generic_transformation(min_from, max_from, min_to, max_to, v):
    return ((max_to - min_to)/(max_from - min_from))*(v - max_from) + max_to

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



# TODO implement caches for min and max

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

    def __init__(self, img_list, img_mode, img_size, bw):
        self.mode = img_mode
        self.w = img_size[0]
        self.h = img_size[1]
        self.img = ImageAbstraction._get_img_matrix(self.w, self.h, img_list)
        self.bw = bw
        self.modified_min_max = True

    def _get_img_matrix(w, h, img_list):
        img = []
        for i in range(w):
            img.append([])
            for j in range(h):
                img[i].append(img_list[j * w + i])
        return img

    def _get_max_min(self):
        if self.modified_min_max:
            self.cached_max_value = max_matrix(self.img)
            self.cached_min_value = min_matrix(self.img)
            self.modified_min_max = False
        return (self.cached_max_value, self.cached_min_value)

    def get_image_bytes(self):
        flat_list = []
        if self.bw:
            max_v, min_v = self._get_max_min()
        for j in range(self.h):
            for i in range(self.w):
                if self.mode == 'RGB':
                    flat_list.extend(list(self.img[i][j]))
                else:
                    flat_list.append(int(transform_to_std(min_v, max_v, self.img[i][j])))
        return bytes(flat_list)

    def update_pixel(self, x, y, color):
        if self.bw:
            max_v, min_v = self._get_max_min()
            v = transform_from_std(min_v, max_v, color)
            self.img[x][y] = v

    def get_size_tuple(self):
        return (self.w, self.h)

    def negative(self):
        if self.bw:
            max_v, min_v = self._get_max_min()
            f = lambda x: max_v - x + min_v
        else:
            f = lambda x: tuple(255 - e for e in x)
        self.img = map_matrix(self.img, self.w, self.h, f)
        self.modified_min_max = True

    def umbral(self, value):
        max_v, min_v = self._get_max_min()
        v = transform_from_std(min_v, max_v, value)
        self.img = map_matrix(self.img, self.w, self.h, lambda x: max_v if x > v else min_v)
        self.modified_min_max = True

    def power(self, value):
        max_v, min_v = self._get_max_min()
        self.img = map_matrix(self.img, self.w, self.h, lambda x : (255/pow(max_v, value))*pow(x, value))
        self.modified_min_max = True

    def product(self, value):
        self.img = map_matrix(self.img, self.w, self.h, lambda x : x*value)
        self.modified_min_max = True
