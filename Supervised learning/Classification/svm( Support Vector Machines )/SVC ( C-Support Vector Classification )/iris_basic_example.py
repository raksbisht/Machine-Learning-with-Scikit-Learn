from sklearn import datasets #import the datasets comes with scikit learn
from sklearn import svm      #Import the SVM algo class

iris=datasets.load_iris() #Load the iris datasets from scikit datasets

iris_data=iris.data #iris data 
iris_target=iris.target #iris target data
iris_target_names=iris.target_names #iris target names

#print(iris_data) 
#print(iris_target)
#print(iris_target_names)

#Now below is the code for svc classifier

clf=svm.SVC() #using svc algo from SVM

clf.fit(iris_data,iris_target) #fit method is used here to training the model 

#clf.fit(iris_data,iris_target_names[iris_target]) #to get the results as names instead of numeric values

predict_data=iris_data[0:1] #for example we are here getting the first record of iris data to test the predict

clf.predict(predict_data) #here predict method is used to predict the result




