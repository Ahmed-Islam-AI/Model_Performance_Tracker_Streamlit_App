# Necessary Libraries
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA  # principal Components algorithm
import matplotlib.pyplot as plt

st.title('Model Performance Tracker 📊🚀')
st.write('''A simple and interactive tool to explore and compare 
         machine learning classifiers on various datasets. 
         Choose your dataset, adjust model parameters, 
         and visualize performance effortlessly.''')


st.subheader("""
         Which one is the best model for the given dataset?""")

# Sidebar drop-down for selecting a built-in Streamlit dataset
datagroup_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

# Sidebar drop-down for model selection
classifier_group_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "RandomForest"))

# Loading datasets from Sklearn library
def pick_dataset(datagroup_name):
    if datagroup_name == 'Iris':
        data = datasets.load_iris()
    elif datagroup_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Calling the function
X, y = pick_dataset(datagroup_name)
st.write(f"Shape of the {datagroup_name} Dataset is {X.shape}")
st.write(f"Num of Classes in {datagroup_name} dataset are {len(np.unique(y))}")

def put_parametes_ui(classifier):
    params = dict()
    if classifier == "KNN":
        k = st.sidebar.slider("K", 1, 14)
        params["k"] = k
    elif classifier == "SVM":
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 1, 20)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

# Calling parameters function  
params = put_parametes_ui(classifier_group_name)

def have_classifier(classifier, params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    elif classifier == "SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=42)
    return clf

clf = have_classifier(classifier_group_name, params)

# Splitting the dataset into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting 
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Checking for the accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_group_name}')
st.write(f'Accuracy = {accuracy*100}')

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader("Scatter Plot")

# Scatter Plot
pca = PCA(2)
X_projection = pca.fit_transform(X)

x1 = X_projection[:, 0]
x2 = X_projection[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# Plotting on Web Page 
st.pyplot(fig)
