import numpy as np 
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt 
import argparse
from trimesh import load_off, save_off

def get_cotan_laplacian(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy
    
    Parameters
    ----------
    VPos : ndarray (N, 3) 
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices 
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float
    

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    #Allocate space for the sparse array storage, with 2 entries for every
    #edge for eves ry triangle (6 entries per triangle); one entry for directed 
    #edge ij and ji.  Note that this means that edges with two incident triangles
    #will have two entries per directed edge, but sparse array will sum them 
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.zeros(M*6)
    
    #Keep track of areas of incident triangles and the number of incident triangles
    IA = np.zeros(M*3)
    VA = np.zeros(M*3) #Incident areas
    VC = 1.0*np.ones(M*3) #Number of incident triangles
    
    #Step 1: Compute cotangent weights
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        dV1 = VPos[ITris[:, i], :] - VPos[ITris[:, k], :]
        dV2 = VPos[ITris[:, j], :] - VPos[ITris[:, k], :]
        Normal = np.cross(dV1, dV2)
        #Cotangent is dot product / mag cross product
        NMag = np.sqrt(np.sum(Normal**2, 1))
        cotAlpha = np.sum(dV1*dV2, 1)/NMag
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        V[shift*M*2:shift*M*2+M] = cotAlpha
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
        V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
        if shift == 0:
            #Compute contribution of this triangle to each of the vertices
            for k in range(3):
                IA[k*M:(k+1)*M] = ITris[:, k]
                VA[k*M:(k+1)*M] = 0.5*NMag
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    #Scale each row by the incident areas TODO: Fix this
    """
    Areas = sparse.coo_matrix((VA, (IA, IA)), shape=(N, N)).tocsr()
    Areas = Areas.todia().data.flatten()
    Areas[Areas == 0] = 1
    Counts = sparse.coo_matrix((VC, (IA, IA)), shape=(N, N)).tocsr()
    Counts = Counts.todia().data.flatten()
    RowScale = sparse.dia_matrix((3*Counts/Areas, 0), L.shape)
    L = L.T.dot(RowScale).T
    """
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def get_umbrella_laplacian(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with "umbrella weights" (unweighted)
    by doing many operations in parallel using NumPy
    
    Parameters
    ----------
    VPos : ndarray (N, 3) 
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices 
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float
    

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with umbrella weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    
    #Step 1: Set up umbrella entries
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    L[L > 0] = 1
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L




def get_laplacian_spectrum(VPos, ITris, K):
    """
    Given a mesh, to compute first K eigenvectors of its Laplacian
    and the corresponding eigenvalues
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvectors to compute
    Returns
    -------
    (eigvalues, eigvectors): a tuple of the eigenvalues and eigenvectors
    """
    L = get_cotan_laplacian(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    return (eigvalues, eigvectors)


def get_heat(eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    """
    Simulate heat flow by projecting initial conditions
    onto the eigenvectors of the Laplacian matrix, and then sum up the heat
    flow of each eigenvector after it's decayed after an amount of time t
    Parameters
    ----------
    eigvalues : ndarray (K)
        Eigenvalues of the laplacian
    eigvectors : ndarray (N, K)
        An NxK matrix of corresponding laplacian eigenvectors
        Number of eigenvectors to compute
    t : float
        The time to simulate heat flow
    initialVertices : ndarray (L)
        indices of the verticies that have an initial amount of heat
    heatValue : float
        The value to put at each of the initial vertices at the beginning of time
    
    Returns
    -------
    heat : ndarray (N) holding heat values at each vertex on the mesh
    """
    N = eigvectors.shape[0]
    I = np.zeros(N)
    I[initialVertices] = heatValue
    coeffs = I[None, :].dot(eigvectors)
    coeffs = coeffs.flatten()
    coeffs = coeffs*np.exp(-eigvalues*t)
    heat = eigvectors.dot(coeffs[:, None])
    return heat

def get_hks(VPos, ITris, K, ts):
    """
    Given a triangle mesh, approximate its curvature at some measurement scale
    by recording the amount of heat that remains at each vertex after a unit impulse
    of heat is applied.  This is called the "Heat Kernel Signature" (HKS)

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvalues/eigenvectors to use
    ts : ndarray (T)
        The time scales at which to compute the HKS
    
    Returns
    -------
    hks : ndarray (N, T)
        A array of the heat kernel signatures at each of N points
        at T time intervals
    """
    L = get_cotan_laplacian(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None]*ts.flatten()[None, None, :])
    return np.sum(res, 1)


def saveHKSColors(filename, VPos, hks, ITris, cmap = 'gray'):
    """
    Save the mesh as a .coff file using a divergent colormap, where
    negative curvature is one one side and positive curvature is on the other
    """
    c = plt.get_cmap(cmap)
    x = (hks - np.min(hks))
    x /= np.max(x)
    np.array(np.round(x*255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    save_off(filename, VPos, C, ITris)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to OFF file for triangle mesh on which to compute the HKS")
    parser.add_argument("--output", type=str, required=True, help="Path to OFF file which holds a colored mesh showing the HKS")
    parser.add_argument("--t", type=float, required=True, help="Time parameter for the HKS")
    parser.add_argument("--neigvecs", type=int, required=False, default = 200, help="Number of eigenvectors to use")

    opt = parser.parse_args()
    (VPos, VColors, ITris) = load_off(opt.input)
    neigvecs = min(VPos.shape[0], opt.neigvecs)
    hks = get_hks(VPos, ITris, neigvecs, np.array([opt.t]))
    saveHKSColors(opt.output, VPos, hks[:, 0], ITris)
