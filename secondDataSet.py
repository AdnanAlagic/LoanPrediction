from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# Reading a data set from git repository
dataset = pd.read_csv(
    r'https://raw.githubusercontent.com/AdnanAlagic/LoanSet2/main/LoanDataset2.csv')

print(dataset)
# Preprocessing data
le = preprocessing.LabelEncoder()

# Dealing with empty fields
# dataset['Income'] = dataset['Income'].fillna(dataset['Income'].mean())
# dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
# dataset['Experience'] = dataset['Experience'].fillna(dataset['Experience'].mean())


# Eliminating NaN or missing input numbers
# df_binary.fillna(method ='ffill', inplace = True)

# dataset.isna().sum() / dataset.shape[0] * 100

# Extracting data set columns
gender = dataset.iloc[:, 1].values
married = dataset.iloc[:, 2].values
education = dataset.iloc[:, 3].values
self_employed = dataset.iloc[:, 4].values
applicant_income = dataset.iloc[:, 5].values
loan_amount = dataset.iloc[:, 6].values
loan_amount_term = dataset.iloc[:, 7].values
credit_history = dataset.iloc[:, 8].values
property_area = dataset.iloc[:, 9].values
loan_status = dataset.iloc[:, 10].values

# Transformation of columns
gender_encoded = le.fit_transform(gender)
married_encoded = le.fit_transform(married)
education_encoded = le.fit_transform(education)
self_employed_encoded = le.fit_transform(self_employed)
applicant_income_encoded = le.fit_transform(applicant_income)
loan_amount_encoded = le.fit_transform(loan_amount)
loan_amount_term_encoded = le.fit_transform(loan_amount_term)
credit_history_encoded = le.fit_transform(credit_history)
property_area_encoded = le.fit_transform(property_area)
loan_status_encoded = le.fit_transform(loan_status)

# Setting model, depending on choosen algorithm
# model = KNeighborsClassifier(n_neighbors=100, p=2, metric='euclidean')
#model = GaussianNB()
model = DecisionTreeClassifier(criterion="entropy")
# model = RandomForestClassifier(n_estimators=100)
# model = LinearRegression()
# model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)


X_train, X_test, y_train, y_test = train_test_split(
    list(zip(gender_encoded, married_encoded, education_encoded, self_employed_encoded,
             applicant_income_encoded, loan_amount_encoded, loan_amount_term_encoded, credit_history_encoded,
             property_area_encoded)),
    loan_status_encoded,
    test_size=0.3, random_state=1)

# Model training
model.fit(list(X_train), y_train)

# Printing model accuracy
y_pred = model.predict(X_test)
print("Model accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Model F1 score: ", f1_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

# Printing confusion matrix in console
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Importing seaborn as sns in order ti display confusion matrix in real image
import seaborn as sns

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten() / np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]

labels = np.asarray(labels).reshape(2, 2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

# Display the visualization of the Confusion Matrix.
plt.show()
