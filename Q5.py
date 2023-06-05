import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
import plotly.graph_objects as go


def load_3d_point_cloud():
    # Load the XYZ coordinates from the OFF file
    with open('tent_0163.off', 'r') as f:
        lines = f.readlines()

    points = []
    for line in lines[1:]:
        if len(line.split()) == 3:
            x, y, z = line.split()
        points.append([float(x), float(y), float(z)])

    # Convert the points list to a numpy array
    points = np.array(points)
    avg = np.mean(points, axis=0)
    return points, avg


def plot_via_matplotlib(points):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def plot_via_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def plot_via_pyvista(points):
    cloud = pv.PolyData(points)
    cloud.plot()

# add noise sampled from N(0,0.01s^2I) to the point cloud
def add_noise(points, s=0.01):
    noise = np.random.normal(0, (0.01 * (s**2)) , points.shape)
    return points + noise


# Construct a sparse affinity matrix by a Gaussian kernel with the Euclidean distance in a
def construct_affinity_matrix(points, r=100):
    # Compute the pairwise Euclidean distances
    distances = np.sqrt(((points[:, None] - points) ** 2).sum(axis=2))

    # Compute the affinity matrix by a Gaussian kernel
    W = np.exp(-distances ** 2 / (2 * r ** 2))
    return W

# Compute the eigendecomposition of the normalized Laplacian matrix and plot the sorted eigenvalues from the smallest to the largest.
def calculate_normalized_laplacian(W):
    # Compute the degree vector
    d = W.sum(axis=1)

    # Compute the Laplacian matrix
    D = np.diag(d)
    L = D - W

    # Compute the normalized Laplacian matrix
    d_sqrt_inv = np.sqrt(np.linalg.inv(D))
    L_norm = d_sqrt_inv.dot(L).dot(d_sqrt_inv)
    return L_norm



# Plot the noisy 3D point cloud colored by first, second, sixth, and tenth eigenvectors.
def plot_3d_point_cloud_with_colors(points, normalized_laplacian, plot=True):
    eig, U = np.linalg.eig(normalized_laplacian)
    eig = np.real(eig)
    idx = eig.argsort()

    eig = eig[idx]
    U = U[:, idx]

    if plot:
        # first eigenvector colorization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=U[:, 0], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # second eigenvector colorization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=U[:, 1], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # sixth eigenvector colorization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=U[:, 5], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # tenth eigenvector colorization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=U[:, 9], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


    return eig

def plot_3d_point_cloud_with_colors_plotly(points, normalized_laplacian, plot=True):
    eig, U = np.linalg.eig(normalized_laplacian)
    eig = np.real(eig)
    idx = eig.argsort()

    eig = eig[idx]
    U = np.real(U[:, idx])

    if plot:
        # first eigenvector colorization
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(color=U[:, 0], size=3, colorscale='Viridis', opacity=0.8)
        ))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

        # second eigenvector colorization
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(color=U[:, 1], size=3, colorscale='Viridis', opacity=0.8)
        ))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

        # sixth eigenvector colorization
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(color=U[:, 5], size=3, colorscale='Viridis', opacity=0.8)
        ))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

        # tenth eigenvector colorization
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(color=U[:, 9], size=3, colorscale='Viridis', opacity=0.8)
        ))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

    return eig

def low_pass_filter(points, lambda_max, tau=1e-4):
    # apply h(x)=exp(-tau*x/lambda_max)
    h = lambda x: np.exp(-tau * x / lambda_max)

    denoised_points = h(points)
    return denoised_points



points, avg = load_3d_point_cloud()
# a
plot_via_pyvista(points)
# b
# noisy_points = add_noise(points, s=np.mean(avg))
noisy_points = add_noise(points, s=15)
plot_via_pyvista(noisy_points)
# plot_via_matplotlib(noisy_points)
# c
W = construct_affinity_matrix(noisy_points, r=np.mean(avg))
L_norm = calculate_normalized_laplacian(W)
# d
eig = plot_3d_point_cloud_with_colors_plotly(noisy_points, L_norm,plot=True)
# e
denoised_points = low_pass_filter(noisy_points, eig[-1])
plot_via_pyvista(denoised_points)
# plot_via_open3d(denoised_points)




