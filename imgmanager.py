from PIL import Image
from bitarray import bitarray

from imgprocessor import ImageAbstraction

import math
import copy

ZOOM_INTENSITY = 50


class ImageManager:

    def __init__(self):
        pass

    def load_rectangle(self, w, h):
        img_list = [0] * (w * h)
        for j in range(int(0.25 * h), int(0.75 * h)):
            for i in range(int(0.25 * w), int(0.75 * w)):
                img_list[j * w + i] = 255
        self.create_images(img_list, 'L', (w, h), True)

    def load_circle(self, w, h):
        r = 0.25 * min(w, h)
        img_list = [0] * (w * h)
        for j in range(h):
            for i in range(w):
                if math.sqrt(pow(j - 0.5 * h, 2) + pow(i - 0.5 * w, 2)) < r:
                    img_list[j * w + i] = 255
        self.create_images(img_list, 'L', (w, h), True)

    def load_image(self, img):
        img_list = list(img.getdata())
        if img.mode == 'RGB':
            if all([e[0] == e[1] == e[2] for e in img_list]):
                img_list = [e[0] for e in img_list]
                bw = True
                mode = 'L'
            else:
                bw = False
                mode = 'RGB'
        elif img.mode == 'L' or img.mode == '1':
            bw = True
            mode = 'L'
        else:
            raise Exception('Unsupported Format')

        self.create_images(img_list, mode, img.size, bw)

    def create_images(self, img_list, mode, img_size, bw):
        self.image = ImageAbstraction(img_list, mode, img_size, bw)
        self.backup = copy.deepcopy(self.image)

        self.cached_backup = Image.frombytes(
            self.backup.mode,
            self.backup.get_size_tuple(),
            self.backup.get_image_bytes())
        self.cached_image = Image.frombytes(
            self.image.mode,
            self.image.get_size_tuple(),
            self.image.get_image_bytes())
        self.modified = False

    def save_image(self, fname):
        self.get_image().save(fname)

    def get_original(self):
        return self.cached_backup

    def get_image(self):
        if self.modified:
            self.cached_image = Image.frombytes(
                self.image.mode,
                self.image.get_size_tuple(),
                self.image.get_image_bytes())
            self.modified = False
        return self.cached_image

    def get_zoomed_original(self, x, y, w, h):
        return ImageManager._get_zoomed_img(self.get_original(), x, y, w, h)

    def get_zoomed_img(self, x, y, w, h):
        return ImageManager._get_zoomed_img(self.get_image(), x, y, w, h)

    def _get_zoomed_img(img, x, y, w, h):
        return img.crop(
            (x - ZOOM_INTENSITY,
             y - ZOOM_INTENSITY,
             x + ZOOM_INTENSITY,
             y + ZOOM_INTENSITY)).resize((w, h))

    def get_img_pixel_color(self, x, y):
        return ImageManager._get_pixel_color(self.image, x, y)

    def get_original_pixel_color(self, x, y):
        return ImageManager._get_pixel_color(self.backup, x, y)

    def _get_pixel_color(img_abstraction, x, y):
        # return in (r,g,b) format
        c = img_abstraction.img[x][y]
        if img_abstraction.bw:
            return (c, c, c)
        return c

    def update_img_pixel(self, x, y, color):
        self.image.update_pixel(x, y, color)
        self.modified = True

    def get_outbound_pixel(self, center_x, center_y, x, y, w, h):
        original_x = int(x / (w / (ZOOM_INTENSITY * 2)))
        original_y = int(y / (h / (ZOOM_INTENSITY * 2)))
        mapped_x = original_x - ZOOM_INTENSITY + center_x
        mapped_y = original_y - ZOOM_INTENSITY + center_y
        return (mapped_x, mapped_y)

    def get_original_selection(self, x1, y1, x2, y2, in_zoom, t=None):
        ((x0, y0), (xf, yf)) = self._map_selection(x1, y1, x2, y2, in_zoom, t)
        return self.get_original().crop((x0, y0, xf, yf))

    def get_studio_selection(self, x1, y1, x2, y2, in_zoom, t=None):
        ((x0, y0), (xf, yf)) = self._map_selection(x1, y1, x2, y2, in_zoom, t)
        return self.get_image().crop((x0, y0, xf, yf))

    def _map_selection(self, x1, y1, x2, y2, in_zoom, t=None):
        if in_zoom:
            (x0, y0) = self.get_outbound_pixel(t[0], t[1], x1, y1, t[2], t[3])
            (xf, yf) = self.get_outbound_pixel(t[0], t[1], x2, y2, t[2], t[3])
        else:
            (x0, y0) = (x1, y1)
            (xf, yf) = (x2, y2)
        return ((x0, y0), (xf, yf))

    def get_statistics(self, img):
        if self.image.bw:
            array = [e for e in img.getdata()]
            return (len(array), round(sum(array) / len(array), 2))
        elif img.mode == 'RGB':
            array = [item for sublist in img.getdata() for item in sublist]
            r, g, b = [], [], []
            for idx, elem in enumerate(array):
                if idx % 3 == 0:
                    r.append(elem)
                elif idx % 3 == 1:
                    g.append(elem)
                elif idx % 3 == 2:
                    b.append(elem)
            l = int(len(array) / 3)
            return (
                l,
                (round(
                    sum(r) / l,
                    2),
                    round(
                    sum(g) / l,
                    2),
                    round(
                    sum(b) / l),
                    2))

    def add_img(self, aux_image_manager):
        self.modified = True
        self.image.add(aux_image_manager.image)

    def negative(self):
        self.modified = True
        self.image.negative()

    def umbral(self, value):
        self.common_operators_on_bw(self.image.umbral, value)

    def power(self, value):
        self.common_operators_on_bw(self.image.power, value)

    def product(self, value):
        self.common_operators_on_bw(self.image.product, value)

    def common_operators_on_bw(self, f, value):
        if self.image.bw and self.image.mode == 'L':
            self.modified = True
            f(value)
        else:
            raise Exception('Unsupported operation')
