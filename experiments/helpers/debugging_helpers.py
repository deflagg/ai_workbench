import io
import matplotlib.pyplot as plt
from PIL import Image

# Display the graph
# This function is used to display the graph of the language chain
# graph: StateGraph object
# name: str
# figure_size: tuple (width, height)
def display_langgraph_graph(graph, name, figure_size=(10, 6)):
    try:
        image_data = graph.get_graph(xray=True).draw_mermaid_png()
        image = Image.open(io.BytesIO(image_data))
        plt.figure(figsize=figure_size)
        plt.imshow(image)      
        plt.axis('off') 
        plt.title(name)
        plt.show()
    except Exception as e:
        # print exception
        print("An error occre: ", e)
        pass