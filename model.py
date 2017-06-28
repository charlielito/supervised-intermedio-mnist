
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
from dataget import data # <== dataget
from sklearn import datasets, svm, metrics


dataset = data("mnist").get()
print("Loading training data...")
features, train_labels = dataset.training_set.arrays()
train_images = features.reshape( (features.shape[0], features.shape[1]*features.shape[2]) )/255.0
print train_images.shape, train_labels.shape

print("Loading test data...")
features, test_labels = dataset.test_set.arrays()
test_images = features.reshape( (features.shape[0], features.shape[1]*features.shape[2]) )/255.0
print test_images.shape, test_labels.shape


param_C = 5
param_gamma = 0.05
classifier = svm.SVC(C=param_C,gamma=param_gamma)

#TRAINING
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(train_images, train_labels)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed time: {}'.format(str(elapsed_time)))

# PREDICT
predicted = classifier.predict(test_images)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted)))

cm = metrics.confusion_matrix(test_labels, predicted)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(test_labels, predicted)))
