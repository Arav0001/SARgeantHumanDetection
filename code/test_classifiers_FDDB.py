import cv2 as cv
import sys
import purge
import time
import random
from NMS import NMS

if len(sys.argv) != 2: raise RuntimeError(f"Number of arguments not suitable: {len(sys.argv)} instead of 1. Argument must contain which classifier is being evaluated [HAAR, LBP]")

CLASSIFIER_TYPES = ['HAAR', 'LBP']
CLASSIFIER_TYPE = sys.argv[1]

if CLASSIFIER_TYPE not in CLASSIFIER_TYPES: raise RuntimeError(f"Argument must contain which classifier is being evaluated [HAAR, LBP], {CLASSIFIER_TYPE} is not valid")

with open('../test_list.txt', 'r') as f:
    TEST_IMAGES = [f'../resources/data_test/{x[:-1]}.jpg' for x in f.readlines()]
    
FOLDS = [[] for _ in range(100)]

_TEST_IMAGES = TEST_IMAGES[:-45]

for i in range(len(_TEST_IMAGES)):
    FOLDS[i % 100].append(_TEST_IMAGES[i])

print(f'Classifier is: {CLASSIFIER_TYPE}\n')

def detectImages():
    with open(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/facesList.txt', 'w') as f:
        detector = cv.CascadeClassifier()
        detector.load(f'../out/iter_4/{CLASSIFIER_TYPE.lower()}/cascade.xml')
        
        purge.purge(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/images')
        
        counter = 1
        
        program_start = time.time()
        
        for image in TEST_IMAGES:
            print(f"\r{round(counter / len(TEST_IMAGES) * 100, 1)}% done", end='\r')
            
            f.write(image[23:-4] + '\n')
            orig_img = cv.imread(image)
            img = cv.equalizeHist(cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY))
            img = cv.bilateralFilter(img, 9, 75, 75)
            faces, rejectLevels, confidences = detector.detectMultiScale3(img, outputRejectLevels=True)
            
            if len(faces) > 0:
                try:
                    n_faces, n_confidences = NMS(faces, confidences, 0.2)
                except Exception as e:
                    print(e, image, faces)
                
                f.write(str(len(n_faces)) + '\n')
                
                for face in n_faces:
                    face_i = face
                    face = list(map(float, face))
                    f.write(str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' ' + str(face[3]) + '\n')
                    start = (face_i[0], face_i[1])
                    end = (face_i[2], face_i[3])
                    orig_img = cv.rectangle(orig_img, start, end, (0, 0, 255), 2)
            else:
                f.write(str(len(faces)) + '\n')
            
            path = image[23:].replace('/', '_')
            
            cv.imwrite(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/images/' + path, orig_img)
        
            counter += 1
    
    detections = open(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/detections.txt', 'w')
    
    for i in range(len(FOLDS)):
        TP = 0
        FN = 0
        
        for image in FOLDS[i]:
            orig_img = cv.imread(image)
            img = cv.equalizeHist(cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY))
            img = cv.bilateralFilter(img, 9, 75, 75)
            faces, rejectLevels, confidences = detector.detectMultiScale3(img, outputRejectLevels=True)
            
            if len(faces) > 0:
                try:
                    n_faces, n_confidences = NMS(faces, confidences, 0.2)
                except Exception as e:
                    print(e, image, faces)
                    
                if len(n_faces) > 0: TP += 1
            else: FN += 1
            
        detections.write(str(TP + FN) + '\t' + str(TP) + '\t' + str(FN) + '\n')
    
    detections.close()
            
    program_end = time.time()
    print(f'\n\nTook {round(program_end - program_start, 1)} seconds to complete')
            
    
detectImages()