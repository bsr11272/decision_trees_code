import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


st.session_state.theme = "Dark"

@st.cache_data
def load_data():
    return pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')

def prepare_data(data):
    X = data.drop(['ID', 'Personal Loan'], axis=1)
    y = data['Personal Loan']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def find_best_pruned_tree(X_train, X_val, y_train, y_val):
    results = []
    for max_depth in range(1, 20):  # Adjust range as needed
        for min_samples_split in range(2, 20):  # Adjust range as needed
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            results.append({
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'num_nodes': clf.tree_.node_count,
                'train_error': (1 - train_acc) * 100,  # Convert to percentage
                'val_error': (1 - val_acc) * 100  # Convert to percentage
            })
    return pd.DataFrame(results)

def show_pruned_trees_analysis():
    st.title("Error Rates on Pruned Trees - Bank Loan Data")

    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data)

    df = find_best_pruned_tree(X_train, X_val, y_train, y_val)

    # Create the summary table
    fully_grown = df['num_nodes'].max()
    min_error_tree = df.loc[df['val_error'].idxmin(), 'num_nodes']
    best_pruned = df.loc[(df['num_nodes'] < min_error_tree) & (df['val_error'] <= df['val_error'].min() * 1.05), 'num_nodes'].min()

    summary_df = pd.DataFrame({
        'Tree Type': ['Fully-grown tree', 'Minimum error tree', 'Best pruned tree'],
        'Number of Decision Nodes': [fully_grown, min_error_tree, best_pruned]
    })
    st.table(summary_df)

    # Display the detailed table with highlighting
    st.subheader("Detailed Error Rates")
    display_df = df[['num_nodes', 'train_error', 'val_error']].sort_values('num_nodes', ascending=False)
    display_df.columns = ['# Decision Nodes', '% Error Training', '% Error Validation']
    
    def highlight_rows(row):
        if row['# Decision Nodes'] == fully_grown:
            return ['background-color: #FFA07A'] * len(row)
        elif row['# Decision Nodes'] == min_error_tree:
            return ['background-color: #98FB98'] * len(row)
        elif row['# Decision Nodes'] == best_pruned:
            return ['background-color: #87CEFA'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_rows, axis=1))

    # Create the graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(display_df['# Decision Nodes'], display_df['% Error Training'], label='Training Error', marker='o')
    ax.plot(display_df['# Decision Nodes'], display_df['% Error Validation'], label='Validation Error', marker='o')

    # Highlight specific points
    fully_grown_row = display_df[display_df['# Decision Nodes'] == fully_grown].iloc[0]
    min_err_row = display_df[display_df['# Decision Nodes'] == min_error_tree].iloc[0]
    best_pruned_row = display_df[display_df['# Decision Nodes'] == best_pruned].iloc[0]

    ax.scatter(fully_grown_row['# Decision Nodes'], fully_grown_row['% Error Validation'], color='red', s=100, zorder=5, label='Fully-grown tree')
    ax.scatter(min_err_row['# Decision Nodes'], min_err_row['% Error Validation'], color='green', s=100, zorder=5, label='Min. Err. Tree')
    ax.scatter(best_pruned_row['# Decision Nodes'], best_pruned_row['% Error Validation'], color='blue', s=100, zorder=5, label='Best Pruned Tree')

    ax.set_xlabel('Number of Decision Nodes')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rates vs Number of Decision Nodes')
    ax.legend()
    ax.grid(True)
    ax.invert_xaxis()  # Invert x-axis to show decreasing number of nodes

    st.pyplot(fig)

    # Analysis with colored decision node numbers
    st.subheader("Analysis")
    st.markdown(f"""
    1. **Fully-grown tree**: Has <span style='color: red'>{fully_grown}</span> decision nodes. It has the lowest training error but a higher validation error, indicating potential overfitting.
    
    2. **Minimum Error Tree**: Has <span style='color: green'>{min_error_tree}</span> decision nodes. It achieves the lowest validation error while maintaining a reasonable training error. This represents a good balance between model complexity and generalization.
    
    3. **Best Pruned Tree**: Has <span style='color: blue'>{best_pruned}</span> decision nodes. It shows a slight increase in validation error compared to the Minimum Error Tree, but with a simpler structure. This could be preferred if model simplicity is a priority.
    
    4. **Trend**: As the number of decision nodes decreases:
       - Training error generally increases, as expected with simpler models.
       - Validation error initially decreases, reaching a minimum, then increases again for very simple trees.
       
    5. **Overfitting**: The gap between training and validation errors for the fully-grown tree suggests overfitting. This gap narrows as the tree is pruned.
    
    6. **Optimal Pruning**: The best trade-off between model complexity and performance appears to be around <span style='color: green'>{min_error_tree}</span> nodes (Minimum Error Tree) or <span style='color: blue'>{best_pruned}</span> nodes (Best Pruned Tree), depending on whether you prioritize minimal error or model simplicity.
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_pruned_trees_analysis()