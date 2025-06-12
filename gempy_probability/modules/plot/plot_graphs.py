from io import BytesIO

from matplotlib import image as mpimg, pyplot as plt


def plot_graphviz_to_plt(graph):

    # Render to a PNG image in memory
    img_buffer = BytesIO(graph.pipe(format='png'))

    # Load the image into Matplotlib
    img = mpimg.imread(img_buffer, format='png')

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Remove the axis labels
    ax.axis('off')

    # Show the figure
    plt.show()

