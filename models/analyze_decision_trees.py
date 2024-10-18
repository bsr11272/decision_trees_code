import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def analyze_decision_trees():
    st.title("Decision Tree Model Analysis with Structural Pruning")

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')

    data = load_data()

    # Prepare the data
    X = data.drop(['ID', 'Personal Loan'], axis=1)
    y = data['Personal Loan']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Function to train and evaluate model
    def train_evaluate_model(max_depth, max_leaf_nodes):
        clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'depth': clf.get_depth(),
            'n_leaves': clf.get_n_leaves()
        }

    # Grid search for best parameters
    max_depths = range(1, 20)
    max_leaf_nodes = range(2, 100, 5)
    
    results = []
    for depth in max_depths:
        for leaves in max_leaf_nodes:
            result = train_evaluate_model(depth, leaves)
            results.append({
                'max_depth': depth,
                'max_leaf_nodes': leaves,
                **result
            })

    results_df = pd.DataFrame(results)

    # Find best model
    best_model = results_df.loc[results_df['accuracy'].idxmax()]

    st.subheader("Best Model Parameters")
    st.write(f"Max Depth: {best_model['max_depth']}")
    st.write(f"Max Leaf Nodes: {best_model['max_leaf_nodes']}")
    st.write(f"Accuracy: {best_model['accuracy']:.4f}")
    st.write(f"F1 Score: {best_model['f1']:.4f}")
    st.write(f"Precision: {best_model['precision']:.4f}")
    st.write(f"Recall: {best_model['recall']:.4f}")

    # Plot accuracy vs depth and leaves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    results_df.groupby('max_depth')['accuracy'].mean().plot(ax=ax1)
    ax1.set_xlabel('Max Depth')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Max Depth')

    results_df.groupby('max_leaf_nodes')['accuracy'].mean().plot(ax=ax2)
    ax2.set_xlabel('Max Leaf Nodes')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Max Leaf Nodes')

    st.pyplot(fig)

if __name__ == "__main__":
    analyze_decision_trees()