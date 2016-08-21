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
            self.backup = self.img[:]
        else:
            raise Exception('Unsupported Format')

    def save_image(self, fname):
        self.get_image().save(fname)

    def get_original(self):
        return Image.frombytes(self.mode, self.size, bytes(self.backup))

    def get_image(self):
        return Image.frombytes(self.mode, self.size, bytes(self.img))

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
        self.img.reverse()
