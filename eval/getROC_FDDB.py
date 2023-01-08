from matplotlib import pyplot as plt
import sys

if len(sys.argv) != 2: raise RuntimeError(f"Number of arguments not suitable: {len(sys.argv)} instead of 1. Argument must contain which classifier is the ROC curve being generated for [HAAR, LBP]")

CLASSIFIER_TYPES = ['HAAR', 'LBP']
CLASSIFIER_TYPE = sys.argv[1]

if CLASSIFIER_TYPE not in CLASSIFIER_TYPES: raise RuntimeError(f"Argument must contain which classifier is the ROC curve being generated for [HAAR, LBP], {CLASSIFIER_TYPE} is not valid")

print(f'Classifier is: {CLASSIFIER_TYPE}\n')

if CLASSIFIER_TYPE == 'HAAR':
    fac = 1075
    name = 'Haar'
elif CLASSIFIER_TYPE == 'LBP':
    fac = 3398
    name = 'LBP'

path_ContROC = f"../results/test_results/{CLASSIFIER_TYPE.lower()}/ContROC.txt"
path_DiscROC = f"../results/test_results/{CLASSIFIER_TYPE.lower()}/DiscROC.txt"
path_imgSave = f"../results/test_results/{CLASSIFIER_TYPE.lower()}/ROC.jpg"

# credits to johnny88850tw from https://github.com/johnny88850tw/FDDB-evaluation

# get data
with open(path_DiscROC, 'r') as fp:
    discROC = fp.readlines()

# get disc data x, y
discROC = [line.split() for line in discROC]
disc_x = [float(x[1]) for x in discROC]
disc_y = [float(y[0]) for y in discROC]

font = {'fontname':'Calibri'}

# get data
with open(path_ContROC, 'r') as fp:
    contROC = fp.readlines()

# get disc data x, y
contROC = [line.split() for line in contROC]
cont_x = [float(x[1]) for x in contROC]
cont_y = [float(y[0]) for y in contROC]

### plot data
plt.figure()

# set y limit
plt.ylim((-0,1))
# plt.xlim((0,1))
# print label
plt.xlabel('False Positives', weight='bold', fontsize=14, **font)
plt.ylabel('True Positive Rate (Sensitivity)', weight='bold', fontsize=14, **font)

# plot data
plt.plot(disc_x,disc_y,color = '#0000ff', linewidth = 3.0)
plt.plot(cont_x,cont_y,color = '#ff0000', linewidth = 3.0)

# print data text
plt.title(f'Accuracy Score (Discrete & Continuous) on FDDB Dataset\n({name})', weight='bold', fontsize=16, **font)
plt.text(0.008 * fac, 0.925,'Discrete Score:', weight='bold', fontsize=12, **font)
plt.text(0.008 * fac, 0.875,'Continuous Score:', weight='bold', fontsize=12, **font)
plt.text(0.3 * fac, 0.925, '%.1f' %(disc_y[0] * 100) + '%', color="blue", fontsize=12, **font)
plt.text(0.3 * fac, 0.875, '%.1f' %(cont_y[0] * 100) + '%', color="red", fontsize=12, **font)

# 
plt.grid()

# save img
# plt.figure(figsize=(10, 10))
plt.savefig(path_imgSave)
plt.show()