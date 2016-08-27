def max_matrix(matrix):
    max([max(each) for each in matrix])


def min_matrix(matrix):
    min([min(each) for each in matrix])


def transform(min_v, max_v, v):
    return (255 / (max_v - min_v)) * (v - min_v)


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

    def _get_img_matrix(w, h, img_list):
        img = []
        for i in range(w):
            img.append([])
            for j in range(h):
                img[i].append(img_list[j * w + i])
        return img

    def get_image_bytes(self):
        flat_list = []
        max_v = max(255, max_matrix(self.img))
        min_v = min(0, min_matrix(self.img))
        for j in range(self.h):
            for i in range(self.w):
                if self.mode == 'RGB':
                    flat_list.extend(list(self.img[i][j]))
                else:
                    flat_list.append(transform(max_v, min_v, self.img[i][j]))
        return bytes(flat_list)

    def get_size_tuple(self):
        return (self.w, self.h)

    def negative(self):
        for i in range(self.w):
            if self.bw:
                f = lambda x: 255 - x
            else:
                f = lambda x: tuple(255 - e for e in x)
            self.img[i] = list(map(f, self.img[i]))

    def umbral(self, value):
        for i in range(self.w):
            self.img[i] = list(
                map(lambda x: 255 if x > value else 0, self.img[i]))
