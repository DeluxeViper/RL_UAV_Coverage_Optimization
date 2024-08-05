import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Function to add layers
def add_layer(patches, colors, size=(10, 10), num=1, top_left=[0, 0], loc_diff=[1, -1], grid=False):
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    if grid:  # For non-flattened (2D) input
        for i in range(num):
            for j in range(num):
                patches.append(Rectangle(loc_start + [i*loc_diff[0], j*loc_diff[1]], size[1], size[0]))
                colors.append(0.5 if (i + j) % 2 else 0.7)
    else:
        for ind in range(num):
            patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
            colors.append(0.5 if ind % 2 else 0.7)

# Visualize the Q-Network architecture
def visualize_qnetwork():
    patches = []
    colors = []
    fig, ax = plt.subplots()

    # Q-Network: Input layer (2D) + 3 fully connected layers + Output layer
    size_list = [(5, 5), (20, 20), (20, 20), (5, 5)]  # Different sizes for layers
    num_list = [84, 128, 128, 4]  # Number of neurons or grid size (input layer, fc1, fc2, fc3, output)
    x_diff_list = [0, 150, 250, 250]  # Space between layers
    loc_diff_list = [[1, -1], [1, -1], [1, -1], [1, -1]]  # Spacing of neurons

    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    # Layer names
    layer_names = ["Input Layer\n(84x84 State)", "Fully Connected Layer 1", "Fully Connected Layer 2", "Output Layer\n(Q-Values)"]

    # Visualize layers
    for ind in range(len(size_list)):
        if ind == 0:  # Input layer as 2D grid
            add_layer(patches, colors, size=size_list[ind], num=num_list[ind], top_left=top_left_list[ind], loc_diff=loc_diff_list[ind], grid=True)
        else:
            add_layer(patches, colors, size=size_list[ind], num=num_list[ind], top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        ax.text(top_left_list[ind][0], top_left_list[ind][1] + 30, f'{layer_names[ind]}\n{num_list[ind]} units', fontsize=10)

    # Add patches to the plot
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        patch.set_edgecolor(np.zeros(3))
        ax.add_patch(patch)

    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Run the visualization
visualize_qnetwork()
