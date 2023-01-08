import os
import cv2 as cv
import sys
import purge
import time
from NMS import NMS

if len(sys.argv) != 2: raise RuntimeError(f"Number of arguments not suitable: {len(sys.argv)} instead of 1. Argument must contain which classifier is being evaluated [HAAR, LBP]")

CLASSIFIER_TYPES = ['HAAR', 'LBP']
CLASSIFIER_TYPE = sys.argv[1]

if CLASSIFIER_TYPE not in CLASSIFIER_TYPES: raise RuntimeError(f"Argument must contain which classifier is being evaluated [HAAR, LBP], {CLASSIFIER_TYPE} is not valid")

NEG_IMAGES = os.listdir('../resources/data_negative/')[:-49]
FOLDS = [[] for _ in range(100)]

for i in range(len(NEG_IMAGES)):
    FOLDS[i % 100].append(NEG_IMAGES[i])

print(f'Classifier is: {CLASSIFIER_TYPE}\n')

def detectImages():
    with open(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/negatives.txt', 'w') as f:
        detector = cv.CascadeClassifier()
        detector.load(f'../out/iter_4/{CLASSIFIER_TYPE.lower()}/cascade.xml')
        
        purge.purge(f'../results/test_results/{CLASSIFIER_TYPE.lower()}/images/')
        
        counter = 1
        
        program_start = time.time()
        
        for i in range(len(FOLDS)):
            FP = 0
            TN = 0
            
            for image in FOLDS[i]:
                print(f"\r{round(counter / len(NEG_IMAGES) * 100, 1)}% done", end='\r')
            
                orig_img = cv.imread('../resources/data_negative/' + image)
                img = cv.equalizeHist(cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY))
                img = cv.bilateralFilter(img, 9, 75, 75)
                faces, rejectLevels, confidences = detector.detectMultiScale3(img, outputRejectLevels=True)
                
                if len(faces) > 0:
                    try:
                        n_faces, n_confidences = NMS(faces, confidences, 0.2)
                    except Exception as e:
                        print(e, image, faces)
                        
                    if len(n_faces) > 0: FP += 1
                else: TN += 1
            
                counter += 1
                
            f.write(str(FP + TN) + '\t' + str(FP) + '\t' + str(TN) + '\n')
            
        program_end = time.time()
        
        print(f'\n\nTook {round(program_end - program_start, 1)} seconds to complete')
                    
detectImages()