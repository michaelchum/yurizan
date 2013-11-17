from sklearn.linear_model import SGDRegressor
from numpy import genfromtxt, savetxt
from numpy import shape
import numpy
import dateutil.parser
import math
import scipy as sp
#from sklearn import cross_validation
#from sklearn import datasets
#from sklearn import metrics
#from sklearn import linear_model
#from sklearn import svm
from sklearn import tree
import cPickle
import arrow

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('intro_datafo.csv','r'), dtype=None, delimiter=',')[1:] 
    newdataset = numpy.zeros(dataset.shape[0]*8).reshape(dataset.shape[0],8)
    index = 0
    
    for date in dataset:
        utc = arrow.get(date[0])
    	year = date[0][0:4]
    	month = utc.format('M')
    	day = utc.format('d')
        day1 = utc.format('D')
    	time = date[0][11:13] + date[0][14:16]
    	float(month)
    	float(day)
    	float(time)
    	float(date[1])
    	float(date[3])
    	float(date[5])
    	if math.isnan(date[5]):
    		date[5]=18000

        newdata = [month,day,time,0,date[2],date[3],0,date[5]]
    	newdataset[index] = newdata
    	index += 1

    savetxt('newdataset.csv', newdataset, delimiter=',', fmt='%f')

    testset = genfromtxt(open('sampleinput3.csv','r'), dtype=None, delimiter=',',usecols = (0,1,2,3,4))[1:]
    newtestset = numpy.zeros(testset.shape[0]*7).reshape(testset.shape[0],7)
    index1 = 0
    print testset
    for date in testset:
        utc = arrow.get(date[0])
        year = date[0][0:4]
        month = utc.format('M')
        day = utc.format('d')
        day1 = utc.format('D')
        time = date[0][11:13] + date[0][14:16]
        float(month)
        float(day)
        float(time)
        float(date[1])
        float(date[3])
    	newdata1 = [month,day,time,0,date[2],date[3],0]
    	newtestset[index1] = newdata1
    	index1 += 1

    savetxt('newtestset.csv', newtestset, delimiter=',', fmt='%f')
    target = [x[7] for x in newdataset]
    train = [x[0:7] for x in newdataset]


	#clf = SGDRegressor(loss="epsilon_insensitive")
	#clf = linear_model.Lasso(alpha = 0.1)
	#clf = linear_model.BayesianRidge()
	#clf = linear_model.LassoLars(alpha=.01
    #clf = linear_model.Ridge (alpha = 0.5)
	#clf = svm.LinearSVC()
    clf = tree.DecisionTreeRegressor()
	#clf = svm.SVR()
    #clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10])
	#clf = linear_model.LinearRegression()
    clf.fit(train, target)
    #print clf.score(train, target)
	#print clf.coef_
	#print interpreter_

	#predicted_probs = [[x[4]] for index, x in enumerate(clf.predict(test))]
    predicted_probs = clf.predict(newtestset)
    savetxt('output9.csv', predicted_probs, delimiter=',', fmt='%f')

    # save the classifier
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(clf, fid) 

if __name__=="__main__":
    main()