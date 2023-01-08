import os
import math

# combine description files
def combineDescs():
    with open('../data/FDDB_list.txt', 'w') as test_list:
        for desc in os.listdir('../data/FDDB-folds/'):
            if len(desc) == 16:
                with open('../data/FDDB-folds/' + desc, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        test_list.write(f'../resources/FDDB/{line[:-1]}.jpg\n')

# sine function in degrees
def sinDeg(num):
    return math.sin(num * math.pi / 180)

# cosine function in degrees
def cosDeg(num):
    return math.cos(num * math.pi / 180)

# square a number
def square(num):
    return num * num

# combine folds files into one large file with bounds
def combineBounds():
    test_bounds = open('../data/FDDB_bounds.txt', 'w')
    test_ellipses = open('../data/FDDB_ellipses.txt', 'w')
    
    for desc in os.listdir('../data/FDDB-folds/'):
        if len(desc) == 28:
            with open('../data/FDDB-folds/' + desc, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    isStart = '/' in line
                    
                    if not isStart and len(line) > 3:
                        line = line[:-4]
                        ellipse = list(map(float, line.split(' ')))
                            
                        # half of bound height is sqrt((acos(o))^2 + (bsin(o))^2)
                        # half of bound width is sqrt((asin(o))^2 + (bcos(o))^2)
                        
                        # ellipse[0] = semi major axis
                        # ellipse[1] = semi minor axis
                        # ellipse[2] = rotation angle
                        # ellipse[3] = central x
                        # ellipse[4] = central y
                            
                        half_h = math.sqrt(square(ellipse[0] * cosDeg(ellipse[2])) + square(ellipse[1] * sinDeg(ellipse[2])))
                        half_w = math.sqrt(square(ellipse[0] * sinDeg(ellipse[2])) + square(ellipse[1] * cosDeg(ellipse[2])))
                        
                        start = (str(ellipse[3] - half_w), str(ellipse[4] + half_h))
                        end = (str(ellipse[3] + half_w), str(ellipse[4] - half_h))
                        
                        ellipse = list(map(str, ellipse))
                        
                        test_bounds.write(' ' + start[0] + ' ' + start[1] + ' ' + end[0] + ' ' + end[1])
                        test_ellipses.write(' ' + ellipse[0] + ' ' + ellipse[1] + ' ' + ellipse[2] + ' ' + ellipse[3] + ' ' + ellipse[4])
                    elif isStart:
                        test_bounds.write(f'\n../resources/FDDB/{line[:-1]}.jpg')
                        test_ellipses.write(f'\n../resources/FDDB/{line[:-1]}.jpg')
                    else:
                        test_bounds.write(f' {line[:-1]}')
                        test_ellipses.write(f' {line[:-1]}')
                        
    test_bounds.close()
    test_ellipses.close()
            
# run functions
combineDescs()
combineBounds()