import cv2

class FlagModule:

    def __init__(self):
        """Initialize image attributes"""
        self.imageWidth = 0
        self.imageHeight = 0


    def readFlag(self, image):
        """Reads the given image"""
        self.image = image
        self.imageHeight, self.imageWidth = self.image.shape[0], self.image.shape[1]


    def cropImage(self, image, bbox):
        """Given image and bounding box, cropped image is extracted"""

        topLeft_x    = bbox[0]
        topLeft_y    = bbox[1]
        topRight_x   = bbox[0]+bbox[2]
        bottomLeft_y = bbox[1]+bbox[3]

        if (topLeft_x    > topRight_x      or
            topLeft_y    > bottomLeft_y    or
            topLeft_x    < 0               or
            topLeft_y    < 0               or
            topRight_x   > self.imageWidth or
            bottomLeft_y > self.imageHeight):
            print('Bounding box error: out of image')
        else:
            self.croppedImage = image[topLeft_y:bottomLeft_y, topLeft_x:topRight_x, :]


    def resize(self, originalImage, size):
        """Resize any image to given size"""
        resizedImage = cv2.resize(originalImage, size)
        return resizedImage


