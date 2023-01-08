import os
import cv2 as cv
import purge
import numpy as np

# output path for positive train images
TRAIN_POS_OUT = '../resources/data_train/'
# output path for positive test images
TEST_POS_OUT = '../resources/data_test/'
# output path for negative test images
NEG_OUT = '../resources/data_negative/'

# gamma power
GAMMA = 2.4

UTKFACE = os.listdir('../resources/UTKFace/')
NEGATIVES = os.listdir('../resources/haarcascade-negatives/images/')
NEGATIVE_IMAGES = os.listdir('../resources/Negative Images/')

# strip directory to just filename
def stripDir(dir: str):
    return dir[dir.find('\\')+1:][:dir[dir.find('\\')+1:].find('.')]

# extracting details from directory
def getDetails(string: str):
    string = stripDir(string)
    
    positions = []
    for i in range(len(string)):
        if string[i] == '_': positions.append(i)
    
    string = string[:positions[2]]
    
    return int(string[0:positions[0]]), int(string[positions[0] + 1:positions[1]]), int(string[positions[1] + 1:])

# get demographics of training images
def getTrainingDemographics():
    ages = [0] * 116
    genders = [0] * 2
    races = [0] * 5
    
    for file in UTKFACE:
        age, gender, race = getDetails(file)
        
        ages[age - 1] += 1
        genders[gender] += 1
        races[race] += 1
        
    f = open('../data/details.txt', 'w')
    
    for i in range(116):
        print('Age', str(i + 1) + ': ' + str(ages[i]), file=f)
        
    print('\nMale:', genders[0], file=f)
    print('Female:', genders[1], file=f)
    
    print('\nWhite:', races[0], file=f)
    print('Black:', races[1], file=f)
    print('Asian:', races[2], file=f)
    print('Indian:', races[3], file=f)
    print('Others:', races[4], file=f)
    
    f.close()

# change gamma of image to get low exposure
def changeGamma(image, gamma):
    table = [np.power(x / 255, gamma) * 255 for x in range(256)]
    table = np.round(np.array(table)).astype(np.uint8)
    return cv.LUT(image, table)

# break list into list of lists
def getSublists(list, size):
    return [list[i:i + size] for i in range(0, len(list), size)]

# get rectangles from text file
def getRects(img):
    with open('../data/FDDB_bounds.txt', 'r') as f:
        lines = [i[:-1] for i in f.readlines()]
        for line in lines:
            data = line.split(' ')
            if data[0] == img:
                del data[0]
                del data[0]
                data = list(map(float, data))
                data = list(map(int, data))
                return getSublists(data, 4)
            
# get ellipses from text file
def getEllipses(img):
    with open('../data/FDDB_ellipses.txt', 'r') as f:
        lines = [i[:-1] for i in f.readlines()]
        for line in lines:
            data = line.split(' ')
            if data[0] == img:
                del data[0]
                del data[0]
                data = list(map(float, data))
                data = list(map(int, data))
                return getSublists(data, 5)

# process negative images
def processNegatives():
    purge.purge(NEG_OUT)
    
    with open('../neg.txt', 'w') as f:
        counter = 0
        for neg_img in NEGATIVE_IMAGES:
            neg_img = cv.resize(cv.imread('../resources/Negative Images/' + neg_img, cv.IMREAD_GRAYSCALE), (100, 100))
            cv.imwrite(NEG_OUT + str(counter) + '.jpg', changeGamma(neg_img, GAMMA))
            f.write(NEG_OUT[3:] + str(counter) + '.jpg\n')
            counter += 1
            
        for neg_img in NEGATIVES:
            neg_img = cv.resize(cv.imread('../resources/haarcascade-negatives/images/' + neg_img, cv.IMREAD_GRAYSCALE), (100, 100))
            cv.imwrite(NEG_OUT + str(counter) + '.jpg', changeGamma(neg_img, GAMMA))
            f.write(NEG_OUT[3:] + str(counter) + '.jpg\n')
            counter += 1
    

# processing training images
def processForTraining():
    purge.purge(TRAIN_POS_OUT)
    
    with open('../train_pos.txt', 'w') as f:
        for i in range(len(UTKFACE)):
            img = cv.resize(cv.imread('../resources/UTKFace/' + UTKFACE[i], cv.IMREAD_GRAYSCALE), (100, 100))
            cv.imwrite(TRAIN_POS_OUT + str(i) + '.jpg', changeGamma(img, GAMMA))
            f.write(TRAIN_POS_OUT[3:] + str(i) + '.jpg' + ' 1 0 0 100 100\n')

# processing testing images
def processForTesting():
    os.system(f'find {TEST_POS_OUT} -type f -delete')
    os.system(f'find ../results/ground_truth -type f -delete')
    
    pos = open('../data/FDDB_list.txt', 'r')
    TEST_POSITIVES = [i[:-1] for i in pos.readlines()]
    pos.close()
    
    # postive images
    test_list = open('../test_list.txt', 'w')
    
    for i in range(len(TEST_POSITIVES)):
        img = cv.imread(TEST_POSITIVES[i], cv.IMREAD_GRAYSCALE)
        img = changeGamma(img, GAMMA)
        cv.imwrite(TEST_POS_OUT + TEST_POSITIVES[i][18:], img)

        rects = getRects(TEST_POSITIVES[i])
        
        for rect in rects:
            start = (rect[0], rect[1])
            end = (rect[2], rect[3])
            edited_image = cv.rectangle(img, start, end, (0, 0, 255), 2)
            
        test_list.write(TEST_POSITIVES[i][18:-4] + '\n')
        
        cv.imwrite('../results/ground_truth/' + TEST_POSITIVES[i][18:], edited_image)
            
    test_list.close()
    
    test_ellipses = open('../test_pos.txt', 'w')
    
    for desc in os.listdir('../data/FDDB-folds/'):
        if len(desc) == 28:
            with open('../data/FDDB-folds/' + desc, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    isStart = '/' in line
                    
                    if not isStart and len(line) > 3:
                        line = line[:-3]
                        ellipse = line.split(' ')
                        test_ellipses.write(ellipse[0] + ' ' + ellipse[1] + ' ' + ellipse[2] + ' ' + ellipse[3] + ' ' + ellipse[4] + '\n')
                    elif isStart:
                        test_ellipses.write(f'{line[:-1]}\n')
                    else:
                        test_ellipses.write(f'{line[:-1]}\n')
            
    test_ellipses.close()
            
# process
processNegatives()
processForTraining()
processForTesting()