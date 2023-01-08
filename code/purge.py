import os
import sys

def purge(dir: str):
    for file in os.listdir(dir): os.remove(dir + '/' + file);
    
if __name__ == "__main__":
    if (len(sys.argv) != 3): print("ERROR: Invalid number of arguments passed, " + str(len(sys.argv) - 1) + " passed instead of 2"); quit()
    if (sys.argv[1] == 'test' or sys.argv[1] == 'train') and (sys.argv[2] == 'positive' or sys.argv[2] == 'negative'):
        purge('../res/final_data_' + sys.argv[1] + '/' + sys.argv[2])
    else:
        print("ERROR: Invalid argument passed")
    