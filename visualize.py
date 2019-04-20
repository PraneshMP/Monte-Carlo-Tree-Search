import networkx as nx
i=0
def visualize(G):
    p=nx.drawing.nx_pydot.to_pydot(G)
    global i
    p.write_png(r'C:\Users\Pranesh\Desktop\SEM 8\RL\package\RL1\attachments (1)\example'+str(i)+'.png')
    i+=1