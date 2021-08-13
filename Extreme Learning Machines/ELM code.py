# Import modules
import pandas as pd
import matplotlib.pyplot as plt
from hpelm.elm import ELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Read datafile, in the same directory, using pandas
data = pd.read_excel('rectangular_properties.xlsx')
# Convert the categorical features to string
data['P-D ']=data['P-D '].astype(str)
data['Type of confinement ']=data['Type of confinement '].astype(str)
data['Confinement code ']=data['Confinement code '].astype(str)
data['Failure ']=data['Failure '].astype(str)
# Delete column 'No. ', which index the observations
del data['No. ']
# Split data into predictors and features
y_data = data['Failure ']
X_data = data
del X_data['Failure ']
# Convert the categorical features to dummies (One-Hot-Encoding can alternatively be used)
X_data_dummies = pd.get_dummies(X_data)
y_data_dummies = pd.get_dummies(y_data)
# Data is stored as NumPy arrays
features = X_data_dummies.loc[:,'f\'c (MPa) ':'Confinement code _9']
X = features.values
y = y_data_dummies.values
# Randomize and split observations into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
# Scale independent variables using MinMaxScalar
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
# Build and train ELM classifier
elmk = ELM(110, 3, batch=1000, tprint = 10)
elmk.add_neurons(110, 'lin')     # Add one hidden layer with 110 units
elmk.train(X_scaled_train, y_train, 'c','CV',k=5)  # Train ELM model using 5-fold cross-validation 

#-------------------------------------------Print Output------------------------------------------------------#
pred = elmk.predict(X_scaled_test).argmax(1)
y_true = y_test.argmax(1)
print("Mean squared error: {:.3f}\n".format(e))       # Mean squared error
print("Accuracy: {:.3f}\n".format(accuracy_score(y_true, pred)))    # Model accuracy
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_true, pred)))  # Confusion matrix
print(classification_report(y_true, pred,target_names=["Flexure", "Shear", "Flexure-Shear"]))  # Classification report
print("\n")

#-----------------------------------------Plot ROC and display its AUC --------------------------------#

n_classes = 3 # number of class
# y_score
y_predicted = elmk.predict(X_scaled_test)
y_true = y_test
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
plot_titles = ['Receiver operating characteristic for FLEXURE FAILURE','Receiver operating characteristic for SHEAR FAILURE',\
               'Receiver operating characteristic for FLEXURE-SHEAR FAILURE']

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_titles[i])
    plt.legend(loc="lower right")
    plt.show()
elmk.save("solved_elmk_for_rectangular_column") #Save the final model



APPENDIX B: COMPLETE PYTHON CODE FOR BUILDING AND EVALUATION THE ELM MODEL ON SRCC DATASET

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
from hpelm.elm import ELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Read datafile, in the same directory, using pandas
data = pd.read_excel('spiral_properties.xlsx')

# Convert the qualitative features to string so that pandas will recognise them as categorical variables
data['P-D ']=data['P-D '].astype(str)
data['Configuration ']=data['Configuration '].astype(str)
data['Cross Section ']=data['Cross Section '].astype(str)
data['Failure Type ']=data['Failure Type '].astype(str)
# Delete column 'No. ', which index the observations
del data['No. ']
# Split data into predictors and features
y_data = data['Failure Type ']
X_data = data
del X_data['Failure Type ']
# Convert the categorical features to dummies (One-Hot-Encoding can alternatively be used)
X_data_dummies = pd.get_dummies(X_data)
y_data_dummies = pd.get_dummies(y_data)
# Data is stored as NumPy arrays
features = X_data_dummies.loc[:,'Diameter ':'P-D _3']
X = features.values
y = y_data_dummies.values
# Randomize and split observations into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Scale independent variables using MinMaxScalar
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
# Build and train ELM classifier
elmk = ELM(29, 3, batch=1000, tprint = 10)
elmk.add_neurons(29, 'lin')                # Add first 29 unit hidden layer
elmk.add_neurons(29, 'lin')                # Add second 29 unit hidden layer
e = elmk.train(X_scaled_train, y_train, 'c','CV',k=5) # Train ELM model using 5-fold cross-validation 

#-------------------------------------------Print Output-------------------------------------------------------#
pred = elmk.predict(X_scaled_test).argmax(1)
y_true = y_test.argmax(1)
print("Mean squared error: {:.3f}\n".format(e))       # Mean squared error
print("Accuracy: {:.3f}\n".format(accuracy_score(y_true, pred)))    # Model accuracy
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_true, pred)))  # Confusion matrix
print(classification_report(y_true, pred,target_names=["Flexure", "Shear", "Flexure-Shear"]))  # Classification report
print("\n")


#-----------------------------------------Plot ROC and display its AUC ----------------------------------------#
n_classes = 3 # number of class
# y_score
y_predicted = elmk.predict(X_scaled_test)
y_true = y_test
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
plot_titles = ['Receiver operating characteristic for FLEXURE FAILURE','Receiver operating characteristic for SHEAR FAILURE',\
               'Receiver operating characteristic for FLEXURE-SHEAR FAILURE']

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_titles[i])
    plt.legend(loc="lower right")
    plt.show()
elmk.save("solved_elmk_for_spiral")  # Save the model
