import cv2

STDIMAGES = "../standard_images/"

class StandardImages:
    def __init__(self, stdimages=STDIMAGES):
        # color
        self.lenna = cv2.imread(stdimages + "color/lenna_color_512.tif", cv2.IMREAD_COLOR)
        self.mandril = cv2.imread(stdimages + "color/mandril_color.tif", cv2.IMREAD_COLOR)
        self.peppers = cv2.imread(stdimages + "color/peppers_color.tif", cv2.IMREAD_COLOR)
        self.airplaneF16 = cv2.imread(stdimages + "color/airplaneF16.tif", cv2.IMREAD_COLOR)
        self.lake = cv2.imread(stdimages + "color/lake_color.tif", cv2.IMREAD_COLOR)
        self.barbara = cv2.imread(stdimages + "color/barbara.tif", cv2.IMREAD_COLOR)
        self.lighthouse = cv2.imread(stdimages + "color/lighthouse.tif", cv2.IMREAD_COLOR)

        # gray
        self.cameraman = cv2.imread(stdimages + "gray/cameraman.tif", cv2.IMREAD_GRAYSCALE)
        self.house = cv2.imread(stdimages + "gray/house.tif", cv2.IMREAD_GRAYSCALE)
        self.jetplane = cv2.imread(stdimages + "gray/jetplane.tif", cv2.IMREAD_GRAYSCALE)
        self.lake_gray = cv2.imread(stdimages + "gray/lake.tif", cv2.IMREAD_GRAYSCALE)
