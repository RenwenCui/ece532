### Linear Ridge Regression

```python
#1 Downloading the Data (MNIST)
import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , mnist.data.shape)
# Print to show there are 1797 labels (integers from 0-9)
print("Label Data Shape", mnist.target.shape)

# imaging processing-normalization(for our data, its influence is weak)
# mnist.data=(mnist.data-np.min(mnist.data))/np.max(mnist.data)

#2 Splitting Data into Training and Test Sets (MNIST)
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
train_lbl = np.asarray(train_lbl,'float64')
test_lbl = np.asarray(test_lbl,'float64')

#3 Showing the Images and Labels (MNIST)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)

#3 Showing the Images and Labels (MNIST)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
    
#4 Scikit-learn 4-Step Modeling Pattern (Digits Dataset)
#4 Step 1. Import the model you want to use
from sklearn.linear_model import RidgeClassifierCV
# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it

#4 Step 2. Make an instance of the Model
reg = RidgeClassifierCV(alphas=[1e-5, 2e-4, 4e-4, 6e-4],normalize=True,cv=10)
# reg = RidgeClassifierCV(alphas=[1e-6, 1e-5, 1e-4, 0.5e-4, 1e-3, 1e-2, 1e-1],normalize=True,cv=10)

#4 Step 3. Training the model on the data, storing the information learned from the data 
#Model is learning the relationship between digits and labels
reg.fit(train_img, train_lbl)
predictions = reg.predict(test_img)

#4 Step 4. Predict the labels of new data (new images)
#Uses the information the model learned during the model training process

# Returns a NumPy Array
# Make predictions on entire test data
predictions = reg.predict(test_img)

score = reg.score(test_img, test_lbl)
print(score)

#6 Display Misclassified images with Predicted Labels (MNIST)
import numpy as np 
import matplotlib.pyplot as plt
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
 if label != predict: 
  misclassifiedIndexes.append(index)
  index +=1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
 plt.subplot(1, 5, plotIndex + 1)
 plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
 plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)

#7 Confusion Matrix (Digits Dataset)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(test_lbl, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')
```
### KNN
```python
#1 Downloading the Data (MNIST)
import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , mnist.data.shape)
# Print to show there are 1797 labels (integers from 0-9)
print("Label Data Shape", mnist.target.shape)

#2 Splitting Data into Training and Test Sets (MNIST)
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
train_lbl = np.asarray(train_lbl,'float64')
test_lbl = np.asarray(test_lbl,'float64')


#3 Showing the Images and Labels (MNIST)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)
    
#4 Scikit-learn 4-Step Modeling Pattern (Digits Dataset)
#4 Step 1. Import the model you want to use
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 20, 2)
accuracies = []

# cross validation: loop over various values of `k` for the k-Nearest Neighbor classifier
for k in kVals:
    model = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
    scores = cross_val_score(model, train_img, train_lbl, cv=10)
    scores_mean = np.mean(scores)
    print("k=%d, accuracy=%.2f%%" % (k, scores_mean * 100))
    accuracies.append(scores_mean)

i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))

#4 Step 2. Make an instance of the Model
knn = neighbors.KNeighborsClassifier(n_neighbors=kVals[i])

#4 Step 3. Training the model on the data, storing the information learned from the data 
#Model is learning the relationship between digits and labels
knn.fit(train_img, train_lbl)

#4 Step 4. Predict the labels of new data (new images)
# evalution
predictions = knn.predict(test_img)
score = knn.score(test_img, test_lbl)
print(score)

#plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(test_lbl, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')
```
