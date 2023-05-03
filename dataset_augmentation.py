import preprocessing
from random import sample
import os
from shutil import rmtree
import cv2 as cv


class datasetAugmentationClass:

    def __init__(self, X_in, transformations, augmented_dataset_size, dataset_dir):
        self.X_in = X_in
        self.operations = transformations
        self.augmented_dataset_size = augmented_dataset_size
        self.dataset_dir = dataset_dir

    def transform_images(self):

        m = len(self.X_in)
        N = int(self.augmented_dataset_size * m)
        if N == 0:
            N = 1

        self.X_out = []
        sampling_distribution = sample(range(m),N)
        for i, idx in enumerate(sampling_distribution):
            img_transformer = preprocessing.ImageTransformer(self.operations)
            self.X_out.append(img_transformer.launch_transform_operation(self.X_in[idx]))

    def export_augmented_dataset(self):

        # Check if directory exists; if so, delete it. Then create a new one
        dataset_dir = os.path.basename(self.dataset_dir)
        for transformation, value in self.operations.items():
            dataset_dir = dataset_dir + '_' + transformation + '_' + str(value[0])
        dataset_dir = os.path.join(os.path.dirname(self.dataset_dir),dataset_dir)
        if os.path.exists(dataset_dir):
            rmtree(dataset_dir)
        os.mkdir(dataset_dir)

        image_basename = 'transformed_image'
        for transformation, value in self.operations.items():
            image_basename = image_basename + '_' + transformation + '_' + str(value[0])

        for i,image in enumerate(self.X_out):
            filename = os.path.join(dataset_dir,'{}_{}.jpg'.format(image_basename,(i+1)))
            cv.imwrite(filename,image)
