from sklearn.linear_model import SGDRegressor
from numpy import genfromtxt, savetxt
from numpy import shape
import numpy
import dateutil.parser
import math
import scipy as sp
import cPickle
import os
import arrow
import csv
from sklearn import tree
from interpolator import interpolate
#from sklearn import cross_validation
#from sklearn import datasets
#from sklearn import metrics
#from sklearn import linear_model
#from sklearn import svm


def decipher(filename):
    intername = interpolate(filename)
    PROJECT_PATH = os.getcwd()
    clf = tree.DecisionTreeRegressor()
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    newfile = "newfile.csv"
    try:
        os.remove(newfile)
    except OSError:
        pass

    f = open(filename, 'rU')
    reader = csv.reader(f, dialect=csv.excel_tab)
    data = [row for row in reader]
    f.close()

    f = open(newfile, "wb")
    writer = csv.writer(f, dialect=csv.excel_tab)

    for x in data:
        if (x!=data[0]):
            writer.writerow(x)

    f.close()

    testset = genfromtxt(open(intername,'r'), dtype=None, delimiter=',',usecols = (0,1,2,3,4,5))
    dateset = genfromtxt(open(newfile,'r'), dtype=None, delimiter=',',usecols = (0,5))

    index = 0
    for date in testset:
        if(date[5]=='null' or date[5]=='' or math.isnan(date[5])):
            index += 1

    newtestset = numpy.zeros(index*7).reshape(index,7)
    isodate = [0 for x in range(index)]

    index1 = 0
    for date in testset:
        
        if(date[5]=='null' or date[5]=='' or math.isnan(date[5])):
            realdate = dateset[index1][0]
            utc = arrow.get(realdate)
            year = realdate[0:4]
            month = utc.format('M')
            day = utc.format('d')
            day1 = utc.format('D')
            time = realdate[11:13] + realdate[14:16]
            print time
            float(month)
            float(day)
            float(time)
            float(date[1])
            float(date[3])
            newdata1 = [month,day,time,0,date[2],date[3],0]
            newtestset[index1] = newdata1
            isodate[index1] = realdate
            index1+=1
    #predicted_probs = [[x[4]] for index, x in enumerate(clf.predict(test))]
    predicted_probs = clf.predict(newtestset)
    output = [["" for i in range(2)] for j in range(index)]
    for i in range(index):
        output[i][0] = isodate[i]
        print output[i][0]
        output[i][1] = predicted_probs[i]

    try:
        os.remove('output.txt')
    except OSError:
        pass

    #savetxt('output.text', output, delimiter=',', fmt='%s,%s')
    savetxt('output.txt', output, delimiter=',', fmt='%s')

    return os.path.join(PROJECT_PATH, 'output.txt')
    
if __name__ == '__main__':
    decipher("sample_input.csv")


