import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.colors as mcolors

# Set page to wide mode
st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('./data/RidingMowers.csv')

data = load_data()

# Create a top navigation bar
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button('Decision Tree'):
        st.query_params['page'] = 'DecisionTree'
with col2:
    if st.button('Fully Grown Decision Tree'):
        st.query_params['page'] = 'model1'
with col3:
    if st.button('Best Pruned Tree'):
        st.query_params['page'] = 'model2'
with col4:
    if st.button('Model Comparison'):
        st.query_params['page'] = 'comparision'
with col5:
    if st.button('Pruned Trees Analysis'):
        st.query_params['page'] = 'analysis'

# Access the current query parameters
params = st.query_params
page = params.get('page', 'DecisionTree')  # Default to 'DecisionTree'

def show_decision_tree():
    st.title("Decision Tree Visualization")

    # Control options
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_feature = st.selectbox("Select Initial Split Feature", ['Income', 'Lot_Size'])
    with col2:
        max_depth = st.slider("Select Tree Depth", min_value=1, max_value=4, value=1)
    with col3:
        if initial_feature == 'Income':
            initial_threshold = st.slider(f"Select {initial_feature} Threshold ($1000s)", 
                                        min_value=int(data[initial_feature].min()), 
                                        max_value=int(data[initial_feature].max()), 
                                        value=60)
        else:
            initial_threshold = st.slider(f"Select {initial_feature} Threshold (1000 sqft)", 
                                        min_value=float(data[initial_feature].min()), 
                                        max_value=float(data[initial_feature].max()), 
                                        value=float(data[initial_feature].mean()))

    # Function to create a logically consistent decision tree
    def create_consistent_tree(X, y, initial_feature, initial_threshold, max_depth):
        feature_order = list(X.columns)
        
        def split(X, y, feature, threshold):
            left_mask = X[feature] <= threshold
            return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

        def build_tree(X, y, depth=0):
            node = {}
            node['samples'] = len(y)
            node['value'] = {'Owner': (y == 'Owner').sum(), 'Nonowner': (y == 'Nonowner').sum()}
            node['depth'] = depth

            if depth == max_depth or node['samples'] < 2 or node['value']['Owner'] == node['samples'] or node['value']['Nonowner'] == node['samples']:
                return node
            
            if depth == 0:
                split_feature = initial_feature
                split_threshold = initial_threshold
            else:
                split_feature = feature_order[depth % len(feature_order)]
                split_threshold = X[split_feature].median()
            
            left_X, left_y, right_X, right_y = split(X, y, split_feature, split_threshold)
            
            node['feature'] = split_feature
            node['threshold'] = split_threshold
            node['left'] = build_tree(left_X, left_y, depth + 1)
            node['right'] = build_tree(right_X, right_y, depth + 1)
            
            return node
        
        return build_tree(X, y)

    # Improved function to plot split lines
    def plot_split_lines(node, ax, x_min, x_max, y_min, y_max):
        if 'feature' in node:
            color = plt.cm.viridis(node['depth'] / max_depth)
            if node['feature'] == 'Income':
                ax.axvline(x=node['threshold'], ymin=(y_min-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                           ymax=(y_max-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                           color=color, linestyle='-', linewidth=2)
                plot_split_lines(node['left'], ax, x_min, node['threshold'], y_min, y_max)
                plot_split_lines(node['right'], ax, node['threshold'], x_max, y_min, y_max)
            else:  # Lot_Size
                ax.axhline(y=node['threshold'], xmin=(x_min-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
                           xmax=(x_max-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
                           color=color, linestyle='-', linewidth=2)
                plot_split_lines(node['left'], ax, x_min, x_max, y_min, node['threshold'])
                plot_split_lines(node['right'], ax, x_min, x_max, node['threshold'], y_max)

    # Updated function to plot the custom tree
    def plot_custom_tree(node, ax, x=0.5, y=0.9, width=0.4):
        if 'feature' in node:
            color = plt.cm.viridis(node['depth'] / max_depth)
            ellipse = Ellipse((x, y), width=width, height=0.1, facecolor=color, edgecolor='black')
            ax.add_patch(ellipse)
            ax.text(x, y, f"{node['feature']}\n<= {node['threshold']:.1f}\nsamples = {node['samples']}", 
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Left branch
            ax.plot([x, x-width/2], [y-0.05, y-0.15], 'k-')
            ax.text(x-width/4, y-0.1, 'True', ha='center', va='center', fontsize=8)
            plot_custom_tree(node['left'], ax, x-width/2, y-0.2, width/2)
            
            # Right branch
            ax.plot([x, x+width/2], [y-0.05, y-0.15], 'k-')
            ax.text(x+width/4, y-0.1, 'False', ha='center', va='center', fontsize=8)
            plot_custom_tree(node['right'], ax, x+width/2, y-0.2, width/2)
        else:
            rect = Rectangle((x-width/2, y-0.05), width, 0.1, facecolor='lightgreen', edgecolor='black')
            ax.add_patch(rect)
            owner_count = node['value']['Owner']
            nonowner_count = node['value']['Nonowner']
            ax.text(x, y, f"O: {owner_count}, NO: {nonowner_count}\nsamples = {node['samples']}", 
                    ha='center', va='center', fontsize=10)

    # Prepare the data
    X = data[['Income', 'Lot_Size']]
    y = data['Ownership']

    # Create logically consistent decision tree
    tree = create_consistent_tree(X, y, initial_feature, initial_threshold, max_depth)

    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Visualization")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a color map
        color_map = {'Owner': 'red', 'Nonowner': 'blue'}
        colors = y.map(color_map)
        
        # Handle any NaN values in the color mapping
        colors = colors.fillna('gray')  # Use 'gray' for any undefined categories
        
        scatter = ax.scatter(X['Income'], X['Lot_Size'], c=colors, s=50)
        ax.set_xlabel('Income ($1000s)')
        ax.set_ylabel('Lot Size (1000 sqft)')
        
        # Plot all split lines with improved function
        plot_split_lines(tree, ax, X['Income'].min(), X['Income'].max(), X['Lot_Size'].min(), X['Lot_Size'].max())
        
        # Add a legend for Owner/Nonowner
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in color_map.items()]
        ax.legend(handles=handles, loc='upper left')
        
        # Add colorbar for tree depth
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=max_depth)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Tree Depth', pad=0.1)
        
        st.pyplot(fig)

    with col2:
        st.subheader("Decision Tree")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plot_custom_tree(tree, ax)
        st.pyplot(fig)

    # Display split information
    st.subheader("Split Information")
    def get_split_info(node):
        if 'feature' in node:
            unit = "$1000s" if node['feature'] == 'Income' else "1000 sqft"
            color = mcolors.rgb2hex(plt.cm.viridis(node['depth'] / max_depth))
            st.markdown(f"<font color='{color}'>Depth {node['depth']}: "
                        f"Split at {node['feature']} <= {node['threshold']:.2f} {unit}</font>", 
                        unsafe_allow_html=True)
            get_split_info(node['left'])
            get_split_info(node['right'])
        else:
            owner_count = node['value']['Owner']
            nonowner_count = node['value']['Nonowner']
            st.markdown(f"<font color='green'>Leaf node: Owners = {owner_count}, "
                        f"Nonowners = {nonowner_count}, samples = {node['samples']}</font>", 
                        unsafe_allow_html=True)

    get_split_info(tree)

    # Display data info
    st.subheader("Data Information")
    st.write(data['Ownership'].value_counts(dropna=False))
    st.write(data.describe())

# Import other model functions
from models.model1 import show_model1
from models.model2 import show_model2
from models.analyze_decision_trees import analyze_decision_trees
from models.show_pruned_trees_analysis import show_pruned_trees_analysis
# Import other models as needed

# Map page names to functions
page_functions = {
    'DecisionTree': show_decision_tree,
    'model1': show_model1,
    'model2': show_model2,
    'comparision': analyze_decision_trees,
    'analysis': show_pruned_trees_analysis,
    # Add other models here
}

# Display the appropriate page based on URL parameter
if page in page_functions:
    page_functions[page]()
else:
    st.error("Page not found! Please check the URL parameters.")