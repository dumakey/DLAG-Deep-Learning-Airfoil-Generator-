import cv2 as cv
import numpy as np


class ImageTransformer:

    def __init__(self, transformations):
        self.transformations = transformations

    @staticmethod
    def resize(frame, resizing=None, plot=False):
        height_old, width_old = frame.shape

        if resizing:
            if type(resizing) == tuple:
                height = resizing[1]
                width = resizing[0]
            elif type(resizing) == float:
                height = resizing * height_old
                width = resizing * width_old

        scale = height / height_old
        if scale < 1:
            new_frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        else:
            new_frame = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)

        if plot == True:
            cv.imshow('Resized', new_frame)
            cv.waitKey(0)

        return new_frame

    def crop(self, image, px_crop=None, plot=False):

        if px_crop:
            px_w0, px_wf, px_h0, px_hf = px_crop
        else:
            px_h_left = np.where(image[:,0,] != 0)
            px_h_right = np.where(image[:,-1,] != 0)
            if px_h_left[0].size != 0:
                px_h0 = np.min([px_h_left[0][0],px_h_right[0][0]])
                px_hf = np.max([px_h_left[0][-1],px_h_right[0][-1]])
            else:
                px_h0 = 0
                px_hf = image.shape[0] - 1

            px_w_top = np.where(image[0,:,] != 0)
            px_w_bottom = np.where(image[-1,:,] != 0)
            if px_w_top[0].size != 0:
                px_w0 = np.min([px_w_top[0][0],px_w_bottom[0][0]])
                px_wf = np.max([px_w_top[0][-1],px_w_bottom[0][-1]])
            else:
                px_w0 = 0
                px_wf = image.shape[1] - 1

        cropped_image = image[px_h0:px_hf,px_w0:px_wf,]

        if plot == True:
            cv.imshow('Cropped', cropped_image)
            cv.waitKey(0)

        return cropped_image

    def rotate(self, image, angle=None, rot_center_x=None, rot_center_y=None, plot=False):

        image_height, image_width = image.shape

        if rot_center_x == None and rot_center_y == None:
            rotCenter = (self.transformations['rotation'][0], self.transformations['rotation'][1])
            if np.any(np.array(rotCenter) == None):
                rotCenter = (image_width//2,image_height//2)
        else:
            rotCenter = (rot_center_x,rot_center_y)

        if angle == None:
            theta = self.transformations['rotation'][-1]
        else:
            theta = angle

        A = cv.getRotationMatrix2D(rotCenter,-theta,scale=1.0)
        rotated_image = cv.warpAffine(image,A,(image_width,image_height))

        rotated_image = self.crop(rotated_image,plot=False)  # remove black regions due to rotation
        rotated_image = self.resize(rotated_image,(image.shape[1],image.shape[0]))

        if plot == True:
            cv.imshow('Rotated', rotated_image)
            cv.waitKey(0)

        return rotated_image

    def translate(self, image, Dx=None, Dy=None, plot=False):

        image_height, image_width = image.shape

        if Dx == None and Dy == None:
            DX, DY = self.transformations['translation']
        else:
            DX = Dx
            DY = Dy
        A = np.array([[1,0,DX],[0,1,DY]],dtype=np.float64)
        translated_image = cv.warpAffine(image,A,(image_width,image_height))

        if plot == True:
            cv.imshow('Translated', translated_image)
            cv.waitKey(0)

        return translated_image

    def flip(self, image, axis):

        if axis == 'horizontal':
            flipped_image = np.flip(image,axis=0)
        elif axis == 'vertical':
            flipped_image = np.flip(image, axis=1)

        return flipped_image

    def filter(self, image, kernel_name=None, kernel_parameters=None, plot=False):

        if kernel_name == None:
            kernel_name = self.transformations['blur']['kernel']
            if kernel_name in ['gaussian','median','bilateral']:
                kernel_parameters = self.transformations['blur']['parameters']
            else:
                kernel_parameters = None

        if kernel_name == 'gaussian':
            kernel_size = kernel_parameters['size']  # kernel size
            sigma = kernel_parameters['sigma']  # kernel sigma

            filtered_image = cv.GaussianBlur(image,(kernel_size,kernel_size),sigma)
        elif kernel_name == 'bilateral':
            smoothing_diameter = kernel_parameters['d']
            sigmaColor = kernel_parameters['sigmaColor']
            sigmaSpace = kernel_parameters['sigmaSpace']

            filtered_image = cv.bilateralFilter(image,smoothing_diameter,sigmaColor,sigmaSpace)
        elif kernel_name == 'median':
            kernel_size = kernel_parameters['ksize']
            filtered_image = cv.medianBlur(image,kernel_size)
        elif kernel_name == 'sharpen':
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            filtered_image = cv.filter2D(image,-1,kernel)
        elif kernel_name == 'emboss':
            kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
            filtered_image = cv.filter2D(image,-1,kernel)
        elif kernel_name == 'sobel':
            kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            filtered_image = cv.filter2D(image,-1,kernel)
        elif kernel_name == 'sepia':
            pass
            #kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
            #filtered_image = cv.filter2D(image,-1,kernel)

        if plot == True:
            cv.imshow('Blur', blur_image)
            cv.waitKey(0)

        return filtered_image

    def zoom(self, image, f, plot=False):

        dims = image.shape
        if f == None:
            zoom_factor = self.transformations['zoom']
        else:
            zoom_factor = f
        angle = 0

        cy, cx = [i//2 for i in dims]
        A = cv.getRotationMatrix2D((cx,cy),angle,zoom_factor)
        zoomed_image = cv.warpAffine(image,A,dims[1::-1],flags=cv.INTER_LINEAR)

        if plot == True:
            cv.imshow('Zoomed image', zoomed_image)
            cv.waitKey(0)

        return zoomed_image

    def launch_transform_operation(self, image):

        operation_functions = {
            'resizing': self.resize,
            'rotation': self.rotate,
            'translation': self.translate,
            'filter': self.filter,
            'zoom': self.zoom,
            'flip': self.flip,
        }

        for ID in self.transformations.keys():
            F = operation_functions[ID]
            fun_args = self.transformations[ID]
            image = F(image,*fun_args)

        return image
