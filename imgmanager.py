from PIL import Image
from bitarray import bitarray
from functools import partial

from imgprocessor import ImageAbstraction, BWImageAbstraction, RGBImageAbstraction

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

    def load_black(self, w, h):
        img_list = [0] * (w * h)
        img_list[0] = 1
        self.create_images(img_list, 'L', (w, h), True)

    def load_white(self, w, h):
        img_list = [255] * (w * h)
        img_list[0] = 254
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
        if mode == 'L':
            self.image = BWImageAbstraction(img_list, img_size)
        elif mode == 'RGB':
            self.image = RGBImageAbstraction(img_list, img_size)

        self.backup = copy.deepcopy(self.image)
        self.undo_list = []

        self.cached_backup = Image.frombytes(
            self.backup.get_mode(),
            self.backup.get_size_tuple(),
            self.backup.get_image_bytes())
        self.cached_image = Image.frombytes(
            self.image.get_mode(),
            self.image.get_size_tuple(),
            self.image.get_image_bytes())
        self.modified = False

    def get_image_width(self):
        if self.has_img():
            return self.image.get_size_tuple()[0]
        raise Exception()

    def get_image_height(self):
        if self.has_img():
            return self.image.get_size_tuple()[1]
        raise Exception()

    def has_img(self):
        return hasattr(self, 'image')

    def save_image(self, fname):
        self.get_image().save(fname)

    def get_original(self):
        return self.cached_backup

    def get_image(self):
        if self.modified:
            self.cached_image = Image.frombytes(
                self.image.get_mode(),
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
        return img_abstraction.get_pixel_color(x, y)

    def update_img_pixel(self, x, y, color):
        self.image.update_pixel(x, y, color)
        self.modify()

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
        if self.image.is_bw():
            array = [e for e in img.getdata()]
            return (len(array), round(sum(array) / len(array), 2))
        elif img.get_mode() == 'RGB':
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

    def get_histogram_values(self):
        return self.image.get_image_list()

    def modify(self):
        self.modified = True
        self.undo_list.append(copy.deepcopy(self.image))

    def to_bw(self):
        self.modify()
        self.image = self.image.get_bw_img()

    def add_img(self, aux_image_manager):
        self.modify()
        self.image.add(aux_image_manager.image)

    def undo(self):
        if not self.undo_list:
            raise Exception()
        self.image = self.undo_list.pop()
        self.modified = True

    def substract_img(self, aux_image_manager):
        self.modify()
        self.image.substract(aux_image_manager.image)

    def multiply_img(self, aux_image_manager):
        self.modify()
        self.image.multiply(aux_image_manager.image)

    def negative(self):
        self.modify()
        self.undo_list.append(copy.deepcopy(self.image))
        self.image.negative()

    def umbral(self, value):
        self.modify()
        self.common_operators_on_bw(self.image.umbral, value)

    def get_global_umbral(self):
        return self.image.get_global_umbral()

    def power(self, value):
        self.modify()
        self.common_operators_on_bw(self.image.power, value)

    def product(self, value):
        self.modify()
        self.common_operators_on_bw(self.image.product, value)

    def compression(self):
        self.modify()
        self.image.compress()

    def enhance_contrast(self, r1, r2):
        self.modify()
        self.image.enhance_contrast(r1, r2)

    def equalize(self):
        self.modify()
        self.image.equalize()

    def common_operators_on_bw(self, f, value):
        if self.image.is_bw() and self.image.get_mode() == 'L':
            f(value)
        else:
            raise Exception('Unsupported operation')

    def exponential_generator(num, param):
        return -(1 / param) * math.log(num)

    def rayleigh_generator(num, param):
        return param * math.sqrt(-2 * math.log(1 - num))

    def exponential_noise(self, param, percentage):
        self.modify()
        self.image.contaminate_multiplicative_noise(
            percentage, partial(ImageManager.exponential_generator, param=param))

    def rayleigh_noise(self, param, percentage):
        self.modify()
        self.image.contaminate_multiplicative_noise(
            percentage, partial(ImageManager.rayleigh_generator, param=param))

    def gauss_noise(self, intensity, percentage):
        self.modify()
        self.image.contaminate_gauss_noise(percentage, intensity)

    def salt_pepper_noise(self, p0, p1, percentage):
        self.modify()
        self.image.contaminate_salt_pepper_noise(percentage, p0, p1)

    def mean_filter(self, size):
        self.modify()
        self.image.mean_filter(size)

    def median_filter(self, size):
        self.modify()
        self.image.median_filter(size)

    def gauss_filter(self, size, sigma):
        self.modify()
        self.image.gauss_filter(size, sigma)

    def border_filter(self, size):
        self.modify()
        self.image.border_filter(size)

    def lorentziano_anisotropic_diffusion(self, sigma, times):
        self.modify()
        self.image.lorentziano_anisotropic_diffusion(sigma, times)

    def leclerc_anisotropic_diffusion(self, sigma, times):
        self.modify()
        self.image.leclerc_anisotropic_diffusion(sigma, times)

    def prewitt_method(self):
        self.modify()
        self.image.prewitt_method()

    def sobel_method(self):
        self.modify()
        self.image.sobel_method()

    def sobel_x_img(self):
        self.modify()
        self.image.sobel_x_to_img()

    def sobel_y_img(self):
        self.modify()
        self.image.sobel_y_to_img()
