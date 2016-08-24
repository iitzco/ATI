from PIL import Image

ZOOM_INTENSITY = 40


class ImageManager:

    def __init__(self):
        pass

    def load_image(self, img):
        self.size = img.size
        if img.mode == 'RGB':
            img_data = img.getdata()
            if all([e[0] == e[1] == e[2] for e in [l for l in img_data]]):
                # means b&w photo
                self.img = [e[0] for e in img_data]
                self.bw = True
                self.mode = 'L'
            else:
                self.img = [x for t in img_data for x in t]
                self.bw = False
                self.mode = 'RGB'
        elif img.mode == 'L':
            self.img = [x for x in img.getdata()]
            self.bw = True
            self.mode = 'L'
        else:
            raise Exception('Unsupported Format')
        self.backup = self.img[:]
        self.cached_backup = Image.frombytes(
            self.mode, self.size, bytes(self.backup))
        self.cached_img = Image.frombytes(
            self.mode, self.size, bytes(self.img))
        self.modified = False

    def save_image(self, fname):
        self.get_image().save(fname)

    def get_original(self):
        return self.cached_backup

    def get_image(self):
        if self.modified:
            self.cached_img = Image.frombytes(
                self.mode, self.size, bytes(self.img))
        self.modified = False
        return self.cached_img

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

    def reverse(self):
        self.modified = True
        self.img.reverse()

    def get_img_pixel_color(self, x, y):
        return ImageManager._get_pixel_color(
            self.img, self.size, self.mode, x, y)

    def get_original_pixel_color(self, x, y):
        return ImageManager._get_pixel_color(
            self.backup, self.size, self.mode, x, y)

    def _get_pixel_color(img, size, mode, x, y):
        pos = y * size[0] + x
        if mode == 'L':
            c = img[pos]
            return (c, c, c)
        elif mode == 'RGB':
            pos = pos * 3
            r = img[pos]
            g = img[pos + 1]
            b = img[pos + 2]
            return (r, g, b)

    def update_img_pixel(self, x, y, color):
        if self.mode == 'L':
            self.img[y * self.size[0] + x] = color
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
