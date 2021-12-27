#import needed packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#reading and filtering data
dataset = pd.read_csv('glacier_land_ice102.csv')
dataset['LST'].fillna(value=dataset['LST'].median(), inplace=True)
dataset.drop(['Unnamed: 0', 'Unnamed: 0.1', 'nbr', 'ndbi', 'ndmi', 'ndsi', 'ndvi', 'ndwi', 'POINTID'], axis=1, inplace=True)

#spliting data into featers and label
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Standalization of data to 0 mean
sc = StandardScaler()
X = sc.fit_transform(X)

#spliting data to 75% trainng and 25% testing with randomization of 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

'''
#testing defrent SVC parameters to choose the optminal parameters using first 1000 data from dataset
param_grid = [{
        'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01]
    }]
optimal_params = GridSearchCV(SVC(), param_grid, scoring='accuracy', verbose=2)
optimal_params.fit(X_train[:1000, :], y_train[:1000])
print(optimal_params.best_params_)
'''

#Biulding SVC classifier and traing it on only 100,000 dataset
linearclassifeir = SVC(kernel='linear', random_state=0, C=10, gamma="scale")
linearclassifeir.fit(X_train[:100000, :], y_train[:100000])

#predicting data of 15,000 data to test the classifier and check for accurecy
predicted = linearclassifeir.predict(X_test[:15000, :])
accuracy = accuracy_score(y_test[:15000], predicted)

#calculating and plot the confiusion matrix for the 15,000 test dataset
matrix = ConfusionMatrixDisplay.from_predictions(y_test[:15000], predicted)
matrix.ax_.set_title('linear SVC Confusion Matrix with accurecy = '+str(accuracy*100))
plt.xlabel('predicted')
plt.ylabel('True')
plt.show()

#calculationg ROC and area under ROC curve
linearauc = roc_auc_score(y_test[:15000], predicted)
linearfpr, lineartpr, threadshot = roc_curve(y_test[:15000], predicted) #linearfpr holds the False Positive rate, lineartpr holds True Positive rate

#plotting the ROC curve
plt.plot(linearfpr, lineartpr,linestyle='--', label='linear SVC with AUC = %0.2f\n' %(linearauc*100))
plt.title('ROC')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend()
plt.show()

#plotting the learning curve
train_sizes, train_scores, test_scores = learning_curve(linearclassifeir, X_test[:1500, :], y_test[:1500], cv=10, scoring='accuracy',n_jobs=-1 ,train_sizes=np.linspace(0.01,1,30) ,verbose=1)

train_mean =np.mean(train_scores , axis=1)
train_std =np.mean(train_scores , axis=1)
test_mean =np.mean(test_scores , axis=1)
test_std =np.mean(test_scores , axis=1)


plt.plot(train_sizes , train_mean ,label='training score')
plt.plot(train_sizes , test_mean ,label='test score')

plt.fill_between(train_sizes-train_std , train_mean+train_std ,color='#ffffff' )
plt.fill_between(train_sizes-test_std , test_mean+test_std ,color='#ffffff' )

plt.title('learning curve')
plt.xlabel('training size')
plt.ylabel('accuracy score')
plt.legend(loc = 'best')
plt.show()