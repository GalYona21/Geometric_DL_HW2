import open3d as o3d
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def calc_L2_dist(points, noisy_points):
    return np.linalg.norm(points - noisy_points)

def get_points(imagenet40_path):
    mesh = o3d.io.read_triangle_mesh(imagenet40_path)
    return np.asarray(mesh.vertices)

def plot_via_pyvista(points):
    cloud = pv.PolyData(points)
    cloud.plot()

def add_noise(points, s=0.5):
    noise = np.random.normal(0, (0.01 * (s ** 2)), points.shape)
    return points + noise

# Construct a sparse affinity matrix by a Gaussian kernel with the Euclidean distance in a
def construct_affinity_matrix(points, r=0.1, sigma=0.1):
    # Compute the pairwise Euclidean distances
    distances = np.sqrt(((points[:, None] - points) ** 2).sum(axis=2))
    mid = np.median(distances.flatten())
    r = mid

    # Initialize affinity matrix with zeros
    affinity_matrix = np.zeros_like(distances)

    # Create the affinity matrix using the distance threshold
    affinity_matrix[distances <= r] = np.exp(-distances[distances <= r] ** 2 / (2 * sigma ** 2))
    #plt.imshow(affinity_matrix)
   # plt.colorbar()
   # plt.show()

    return affinity_matrix

def calculate_normalized_laplacian(affinity_matrix):
    # Compute the degree vector
    d = affinity_matrix.sum(axis=1)

    # Compute the Laplacian matrix
    D = np.diag(d)
    sqrt_D_inv = np.sqrt(np.linalg.inv(D))
    L = D - affinity_matrix

    # Compute the normalized Laplacian matrix
    L_norm = sqrt_D_inv.dot(L).dot(sqrt_D_inv)

    return L_norm

def plot_3d_point_cloud_with_colors(points, normalized_laplacian, plot=True):
    eig, U = np.linalg.eig(normalized_laplacian)
    idx = eig.argsort()
    eig = eig[idx]
    U = U[:, idx]

    eig = np.real(eig)
    U = np.real(U)

    plt.plot(eig)
    plt.title('Eigenvalues')
    plt.show()

    eig = np.round(eig, 5)
    U = np.round(U, 5)
    print(eig)

    np.save('eig.npy', eig)
    np.save('U.npy', U)

    eig = np.load('eig.npy')
    U = np.load('U.npy')

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


    return eig, U

def low_pass_filter(points, eig, U, tau=0.1):
    lambda_max = eig[-1]

    # apply h(x)=exp(-tau*x/lambda_max)
    h = np.exp(-tau * eig / lambda_max)
    H = np.diag(h)
    print(H)

    denoised_points = U.dot(H).dot(U.T).dot(points)
    return denoised_points

points = get_points("person_0022.off")



plot_via_pyvista(points)
noisy_points = add_noise(points)
print("L2_norm before: ", calc_L2_dist(points, noisy_points))
plot_via_pyvista(noisy_points)
affinity_matrix = construct_affinity_matrix(noisy_points)
L_norm = calculate_normalized_laplacian(affinity_matrix)
eig, U = plot_3d_point_cloud_with_colors(noisy_points, L_norm,plot=True)
denoised_points = low_pass_filter(noisy_points, eig, U)
plot_via_pyvista(denoised_points)
print("L2_norm after: ", calc_L2_dist(points, denoised_points))
