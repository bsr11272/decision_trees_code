import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

def show_model1():
    st.title("Fully Grown Decision Tree - Bank Personal Loan Modelling")

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')

    data = load_data()

    # Prepare the data
    X = data.drop(['ID', 'Personal Loan'], axis=1)
    y = data['Personal Loan']

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the fully grown tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Plot the tree
    st.subheader("Fully Grown Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=['No Loan', 'Loan'], filled=True, rounded=True, ax=ax)
    st.pyplot(fig)

    # Function to display confusion matrix and error report
    def display_metrics(y_true, y_pred, dataset_name):
        st.subheader(f"{dataset_name} Data Scoring")
        
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, output_dict=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix:")
            cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            st.dataframe(cm_df)

        with col2:
            st.write("Error Report:")
            error_report = pd.DataFrame({
                'Class': ['0', '1', 'Overall'],
                '#Cases': [cr['0']['support'], cr['1']['support'], cr['macro avg']['support']],
                '#Errors': [
                    cr['0']['support'] - (cm[0, 0] if cm.shape[0] > 1 else cm[0]),
                    cr['1']['support'] - (cm[1, 1] if cm.shape[0] > 1 else 0),
                    sum(y_true != y_pred)
                ],
                '%Error': [
                    (1 - cr['0']['recall']) * 100,
                    (1 - cr['1']['recall']) * 100,
                    (1 - cr['accuracy']) * 100
                ]
            })
            error_report['%Error'] = error_report['%Error'].round(2)
            st.dataframe(error_report)

    # Display metrics for training data
    y_train_pred = clf.predict(X_train)
    display_metrics(y_train, y_train_pred, "Training")

    # Display metrics for validation data
    y_val_pred = clf.predict(X_val)
    display_metrics(y_val, y_val_pred, "Validation")

    # Display metrics for test data
    y_test_pred = clf.predict(X_test)
    display_metrics(y_test, y_test_pred, "Test")

if __name__ == "__main__":
    show_model1()