import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def show_model2():
    st.title("Best Pruned Decision Tree - Bank Personal Loan Modelling")

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

    # Function to find the best pruned tree
    def find_best_pruned_tree(X_train, y_train, X_val, y_val):
        best_score = 0
        best_params = {}
        
        for max_depth in range(1, 20):
            for min_samples_split in range(2, 20):
                clf = DecisionTreeClassifier(max_depth=max_depth, 
                                             min_samples_split=min_samples_split, 
                                             random_state=42)
                clf.fit(X_train, y_train)
                score = clf.score(X_val, y_val)
                if score > best_score:
                    best_score = score
                    best_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split}
        
        best_tree = DecisionTreeClassifier(**best_params, random_state=42)
        best_tree.fit(X_train, y_train)
        return best_tree, best_params

    # Find the best pruned tree
    best_tree, best_params = find_best_pruned_tree(X_train, y_train, X_val, y_val)

    st.write(f"Best pruning parameters: max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")

    # Plot the best pruned tree
    st.subheader("Best Pruned Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_tree, feature_names=X.columns, class_names=['No Loan', 'Loan'], filled=True, rounded=True, ax=ax)
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
    y_train_pred = best_tree.predict(X_train)
    display_metrics(y_train, y_train_pred, "Training")

    # Display metrics for validation data
    y_val_pred = best_tree.predict(X_val)
    display_metrics(y_val, y_val_pred, "Validation")

    # Display metrics for test data
    y_test_pred = best_tree.predict(X_test)
    display_metrics(y_test, y_test_pred, "Test")

    # Display additional information about the tree
    st.subheader("Tree Information")
    st.write(f"Tree Depth: {best_tree.get_depth()}")
    st.write(f"Number of Leaves: {best_tree.get_n_leaves()}")

if __name__ == "__main__":
    show_model2()