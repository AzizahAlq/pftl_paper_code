import pandas as pd
import matplotlib.pyplot as plt

# Data from the image
data = {
    'Number of Clients': [3.0, 4.0, 5.0, 6.0],
    'Average Accuracy (%)': [99.25, 98.54, 98.55, 98.01]
}
df = pd.DataFrame(data)

# Set a smaller global font size for the plot elements
plt.rcParams.update({'font.size': 8})

# Create the plot
plt.figure(figsize=(7, 5))

# Plot the line with markers
plt.plot(
    df['Number of Clients'],
    df['Average Accuracy (%)'],
    color='#e8a31e',  # Golden yellow/orange
    marker='o',
    linestyle='-',
    linewidth=2,
    markersize=8
)

# Set labels and title with specific font sizes
plt.title('Average PFTL Accuracy vs. Number of Clients', fontsize=8)
plt.xlabel('Number of Clients', fontsize=8)
plt.ylabel('Average Accuracy (%)', fontsize=8)

# Set Y-axis limits and ticks
plt.ylim(97.95, 99.3)
plt.yticks([98.0, 98.2, 98.4, 98.6, 98.8, 99.0, 99.2])

# Set X-axis limits and ticks
plt.xlim(2.9, 6.1)
plt.xticks([3.0, 4.0, 5.0, 6.0])

# Add dashed grid lines
plt.grid(True, linestyle='--', alpha=0.6)

# Add annotations for each data point
for i in range(len(df)):
    accuracy = f"{df['Average Accuracy (%)'][i]:.2f}%"
    plt.annotate(
        accuracy,
        (df['Number of Clients'][i], df['Average Accuracy (%)'][i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=10,
        color='black'
    )

# Clean up the appearance
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

 #Save the figure
plt.savefig("recreated_pftl_accuracy_plot.png", bbox_inches='tight')
plt.show() # In a local environment, use this to display