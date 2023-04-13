from keras.models import load_model
import numpy as np
import copy as cp
import cv2 as cv

class VTumorDxModel():
    def __init__(self):
        self.CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        self.tumor_classifier = load_model('static/classification_model/EfficientNetB4_20_10.h5')
        self.tumor_segmenter = load_model('static/segmentation_model/UNET_Segmentation.h5', compile=False)

    def predict_class(self, img):
        img = cp.deepcopy(img)
        img = np.expand_dims(img, axis=0)
        class_pred = self.tumor_classifier.predict(img)
        class_pred = np.argmax(class_pred)
        return self.CLASSES[class_pred]
    
    def tumor_segmentation(self, original_image):
        img = cp.deepcopy(original_image)

        img = img/255
        img = np.expand_dims(img, axis=0)

        segmentation_pred = self.tumor_segmenter.predict(img, verbose=0)[0]
        segmentation_pred = np.squeeze(segmentation_pred, axis=-1)
        segmentation_pred = segmentation_pred >= 0.5
        segmentation_pred = segmentation_pred.astype(np.uint8)

        # Drawing Contours
        contours, hierarchy = cv.findContours(segmentation_pred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        print(len(contours))
        if len(contours) > 0:
            max_score = cv.contourArea(contours[0])
            max_score_index = 0
            for i in range(1, len(contours)):
                if cv.contourArea(contours[i]) > max_score:
                    max_score = cv.contourArea(contours[i])
                    max_score_index = i

            c = contours[max_score_index]
            cv.drawContours(original_image, contours, max_score_index, (0, 0, 255), 2)

        return original_image

