import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# create ring graph with n nodes
def create_graph_ring_laplacian(n):
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    D = np.diag(np.sum(adj, axis=1))
    L = D - adj
    return L



# create the cartesian product of two graphs
def cartesian_product_graph_laplacian(L1, L2):
    n1 = L1.shape[0]
    n2 = L2.shape[0]
    L = np.zeros((n1 * n2, n1 * n2))
    adj = np.zeros((n1 * n2, n1 * n2))
    for i in range(n1):
        for j in range(n2):
            for k in range(n1):
                for l in range(n2):
                    if (L1[i, k] == -1 and j==l) or (L2[j, l] == -1 and i==k):
                        adj[i * n2 + j, k * n2 + l] = 1
                        adj[k * n2 + l, i * n2 + j] = 1


    D = np.diag(np.sum(adj, axis=1))
    L = D - adj
    return L

def compute_eigenvectors(L, k):
    eig, U = np.linalg.eig(L)
    eig = np.real(eig)
    idx = eig.argsort()

    eig = eig[idx]
    U = U[:, idx]
    return eig, U[:, :k]

def plot_graph_topology(eigvecs=None):
    # Create the first ring graph with n nodes
    n = 20
    ring_graph_1 = nx.cycle_graph(n)

    # Create the second ring graph with m nodes
    m = 50
    ring_graph_2 = nx.cycle_graph(m)
    cartesian_product = nx.cartesian_product(ring_graph_1, ring_graph_2)

    pos = nx.spring_layout(cartesian_product)
    fig, ax = plt.subplots()
    if eigvecs is None:
        nx.draw(cartesian_product, ax=ax,pos=pos, node_size=10, width=0.5)
    else:
        nx.draw(cartesian_product, ax=ax, pos=pos, node_size=10, width=0.5, node_color=eigvecs[:, 0])
        ax.set_title("Cartesian Product of Two Ring Graphs Colored by First Eigenvector")
        plt.show()

        fig, ax = plt.subplots()
        nx.draw(cartesian_product, ax=ax, pos=pos, node_size=10, width=0.5, node_color=eigvecs[:, 1])
        ax.set_title("Cartesian Product of Two Ring Graphs Colored by Second Eigenvector")
        plt.show()

        fig, ax = plt.subplots()
        nx.draw(cartesian_product, ax=ax, pos=pos, node_size=10, width=0.5, node_color=eigvecs[:, 5])
        ax.set_title("Cartesian Product of Two Ring Graphs Colored by Sixth Eigenvector")
        plt.show()

        fig, ax = plt.subplots()
        nx.draw(cartesian_product, ax=ax, pos=pos, node_size=10, width=0.5, node_color=eigvecs[:, 9])
        ax.set_title("Cartesian Product of Two Ring Graphs Colored by Tenth Eigenvector")
        plt.show()

    ax.set_title("Cartesian Product of Two Ring Graphs")
    plt.show()

def plot_eig_values(eigs):
    fig, ax = plt.subplots()
    ax.plot(eigs, 'o')
    ax.set_title("Eigenvalues of the Cartesian Product Graph")
    plt.show()



ring_20_laplacian = create_graph_ring_laplacian(20)
ring_50_laplacian = create_graph_ring_laplacian(50)
L = cartesian_product_graph_laplacian(ring_20_laplacian, ring_50_laplacian)
print(L)

# 1
eigs, eig_vecs = compute_eigenvectors(L, 10)


# 2
# a
plot_graph_topology()

# b
plot_eig_values(eigs)

# c
plot_graph_topology(eig_vecs)

# G = nx.Graph()
# G.add_nodes_from(range(20*50))
# # add edges to the cartesian product of G1 and G2
# for i in range(20):
#     for j in range(50):
#         for k in range(20):
#             for l in range(50):
#                 if ring_20_laplacian[i, k] == -1 or ring_50_laplacian[j, l] == -1:
#                     G.add_edge(i * 50 + j, k * 50 + l)
#
#
#
# # Plot graph topology
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# nx.draw(G, ax=ax[1], pos=nx.Graph(G), node_size=10, width=0.1)
# ax[0].set_title("Cartesian Product Graph G1 x G2")
# plt.show()
#
