## ECE532 Final Project
### Dataset

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
#1 Downloading the Data (MNIST)
import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")

print("Image Data Shape" , mnist.data.shape)
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
# reg = RidgeClassifierCV(alphas=[1e-6, 1e-5, 1e-4, 0.5e-4, 1e-3, 1e-2, 1e-1],cv=10)

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
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RenwenCui/ece532/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
