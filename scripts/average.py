from __future__ import print_function
import csv
import sys

#
# generate stats from performance data
#

if (len(sys.argv) < 6) :
    print ("Usage: python stats.py <filename.csv> stats labels cols reps");
    print ("   stats: min max avg mode");
    print ("   labels: number of non-data columns");
    sys.exit();
    
infile = str(sys.argv[1]);
stat = str(sys.argv[2]);
labels = int(sys.argv[3]);
cols = int(sys.argv[4]);
reps = int(sys.argv[5]);

f = open(infile, 'r');

MAXROWS = 1000
data = [[0 for x in range(cols)] for y in range(MAXROWS)]
label = [[0 for x in range(labels)] for y in range(MAXROWS)]

with open(infile) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    rows = 0
    for row in readCSV:
        for j in range(0,labels) :
            label[rows][j] = row[j]
        i = 0
        for j in range(labels,cols + labels) :
            data[rows][i] = float(row[j])
            i = i + 1
        rows = rows + 1

def average() : 
    lb = 0
    ub = reps
    for j in range(0,rows) : 
        for i in range(0,labels) :
            print("%s," % (label[lb][i]), end='');            
        for k in range(0,cols) :
            sum = 0
            for i in range(lb,ub) :
                sum = sum + data[i][k]
            average = sum/reps
            if (k == cols - 1) :             
                print("%3.2f" % (average), end='')     
            else:
                print("%3.2f," % (average), end='')     
        print("")     
        lb = lb + reps
        ub = ub + reps
        if (ub > rows) :
            break

def minmax(max) :
    lb = 0
    ub = reps
    for j in range(0,rows) : 
        for i in range(0,labels) :
            print("%s," % (label[lb][i]), end='');            
        for k in range(0,cols) :
            best = data[lb][k]
            for i in range(lb,ub) :
                if (max == 0) :
                    if (data[i][k] < best) :
                        best = data[i][k]
                else:
                    if (data[i][k] >= best) :
                        best = data[i][k]
            if (k == cols - 1) :             
                print("%3.2f" % (best), end='')     
            else:
                print("%3.2f," % (best), end='')     
        print("")     
        lb = lb + reps
        ub = ub + reps
        if (ub > rows) :
            break
                
def mode() :
    lb = 0
    ub = reps

    for j in range(0,rows) : 
        for i in range(0,labels) :
            print("%s," % (label[lb][i]), end='');            
        for k in range(0,cols) :
            count = [0,0,0,0,0,0,0,0,0,0]
            vals = [0,0,0,0,0,0,0,0,0,0]
            m = 0
            vals[m] = data[lb][k]
            for i in range(lb,ub) :
                n = isIn(data[i][k],vals)
                if (n < 0) :
                    m = m + 1;
                    count[m] = count[m] + 1
                    vals[m] = data[i][k]
                else:
                    count[n] = count[n] + 1

            max = 0
            max_index = 0
            i = 0
            for x in count:
                if (x > max):
                    max = x
                    max_index = i
                i = i + 1
            if (k == cols - 1) :             
                print("%3.2f" % (vals[max_index]), end='')     
            else:
                print("%3.2f," % (vals[max_index]), end='')     

        print("")     
        lb = lb + reps
        ub = ub + reps
        if (ub > rows) :
            break

def isIn(x, vals = []) :
    m = 0
    for a in vals:
        if ((x <= a + 5) and (x >= a - 5)):
          return m
        m = m + 1
    return -1

if (stat == "min") :
    minmax(0)
if (stat == "max") :
    minmax(1)
if (stat == "avg") :
    average()
if (stat == "mode") :
    mode()
    
f.close()



