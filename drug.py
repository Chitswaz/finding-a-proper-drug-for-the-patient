import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree

drug_data= pd.read_csv('drug200.csv')

#data visualization
drug_data.head()

X= drug_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = drug_data["Drug"]

# converting categorical variables to numeric - sex, cholesterol, bp
converter= sklearn.preprocessing.LabelEncoder()
drug_data['Sex']= converter.fit_transform(drug_data['Sex'])
drug_data['Cholesterol']= converter.fit_transform(drug_data['Cholesterol'])
drug_data['BP']= converter.fit_transform(drug_data['BP'])

X= drug_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]

# train test split
X_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(X, y, random_state= 1, test_size= 0.3)


# modelling
model= sklearn.tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 4)
model.fit(X_train, y_train)
y_predicted= model.predict(x_test)

# model evaluation
accuracy= sklearn.metrics.accuracy_score(y_test, y_predicted )
print(accuracy)



