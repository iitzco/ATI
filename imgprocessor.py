from PIL import Image

ZOOM_INTENSITY = 40


class ImageManager:

    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img
        self.backup = img.copy()

    def save_image(self, fname):
        self.img.save(fname)

    def get_original(self):
        return self.backup

    def get_image(self):
        return self.img

    def get_zoomed_original(self, x, y, w, h):
        return ImageManager._get_zoomed_img(self.backup, x, y, w, h)

    def get_zoomed_img(self, x, y, w, h):
        return ImageManager._get_zoomed_img(self.img, x, y, w, h)

    def _get_zoomed_img(img, x, y, w, h):
        return img.crop(
            (x - ZOOM_INTENSITY,
             y - ZOOM_INTENSITY,
             x + ZOOM_INTENSITY,
             y + ZOOM_INTENSITY)).resize((w, h))

    def rotate(self, grade):
        self.img = self.img.rotate(grade)
