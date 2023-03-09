import numpy as np 
from scipy import sparse 


def load_off(filename):
    """
    Load in an OFF file, assuming it's a triangle mesh
    Parameters
    ----------
    filename: string
        Path to file
    Returns
    -------
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    fin = open(filename, 'r')
    nVertices = 0
    nFaces = 0
    lineCount = 0
    face = 0
    vertex = 0
    divideColor = False
    VPos = np.zeros((0, 3))
    VColors = np.zeros((0, 3))
    ITris = np.zeros((0, 3))
    for line in fin:
        lineCount = lineCount+1
        fields = line.split() #Splits whitespace by default
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        #Check section
        if nVertices == 0:
            if fields[0] == "OFF" or fields[0] == "COFF":
                if len(fields) > 2:
                    fields[1:4] = [int(field) for field in fields]
                    [nVertices, nFaces, nEdges] = fields[1:4]
                    #Pre-allocate vertex arrays
                    VPos = np.zeros((nVertices, 3))
                    VColors = np.zeros((nVertices, 3))
                    ITris = np.zeros((nFaces, 3))
                if fields[0] == "COFF":
                    divideColor = True
            else:
                fields[0:3] = [int(field) for field in fields]
                [nVertices, nFaces, nEdges] = fields[0:3]
                VPos = np.zeros((nVertices, 3))
                VColors = np.zeros((nVertices, 3))
                ITris = np.zeros((nFaces, 3))
        elif vertex < nVertices:
            fields = [float(i) for i in fields]
            P = [fields[0],fields[1], fields[2]]
            color = np.array([0.5, 0.5, 0.5]) #Gray by default
            if len(fields) >= 6:
                #There is color information
                if divideColor:
                    color = [float(c)/255.0 for c in fields[3:6]]
                else:
                    color = [float(c) for c in fields[3:6]]
            VPos[vertex, :] = P
            VColors[vertex, :] = color
            vertex = vertex+1
        elif face < nFaces:
            #Assume the vertices are specified in CCW order
            fields = [int(i) for i in fields]
            ITris[face, :] = fields[1:fields[0]+1]
            face = face+1
    fin.close()
    VPos = np.array(VPos, np.float64)
    VColors = np.array(VColors, np.float64)
    ITris = np.array(ITris, np.int32)
    return (VPos, VColors, ITris)

def save_off(filename, VPos, VColors, ITris):
    """
    Save a .off file
    Parameters
    ----------
    filename: string
        Path to which to write .off file
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors.size > 0:
            fout.write(" %g %g %g"%tuple(VColors[i, :]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()
    

def sample_by_area(VPos, ITris, npoints, colPoints = False):
    """
    Randomly sample points by area on a triangle mesh.  This function is
    extremely fast by using broadcasting/numpy operations in lieu of loops

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    npoints : int
        Number of points to sample
    colPoints : boolean (default True)
        Whether the points are along the columns or the rows
    
    Returns
    -------
    (Ps : NDArray (npoints, 3) array of sampled points, 
     Ns : Ndarray (npoints, 3) of normals at those points   )
    """
    ###Step 1: Compute cross product of all face triangles and use to compute
    #areas and normals (very similar to code used to compute vertex normals)

    #Vectors spanning two triangle edges
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    FNormals = np.cross(V1, V2)
    FAreas = np.sqrt(np.sum(FNormals**2, 1)).flatten()

    #Get rid of zero area faces and update points
    ITris = ITris[FAreas > 0, :]
    FNormals = FNormals[FAreas > 0, :]
    FAreas = FAreas[FAreas > 0]
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]

    #Compute normals
    NTris = ITris.shape[0]
    FNormals = FNormals/FAreas[:, None]
    FAreas = 0.5*FAreas
    FNormals = FNormals
    VNormals = np.zeros_like(VPos)
    VAreas = np.zeros(VPos.shape[0])
    for k in range(3):
        VNormals[ITris[:, k], :] += FAreas[:, None]*FNormals
        VAreas[ITris[:, k]] += FAreas
    #Normalize normals
    VAreas[VAreas == 0] = 1
    VNormals = VNormals / VAreas[:, None]

    ###Step 2: Randomly sample points based on areas
    FAreas = FAreas/np.sum(FAreas)
    AreasC = np.cumsum(FAreas)
    samples = np.sort(np.random.rand(npoints))
    #Figure out how many samples there are for each face
    FSamples = np.zeros(NTris, dtype=np.int32)
    fidx = 0
    for s in samples:
        while s > AreasC[fidx]:
            fidx += 1
        FSamples[fidx] += 1
    #Now initialize an array that stores the triangle sample indices
    tidx = np.zeros(npoints, dtype=np.int64)
    idx = 0
    for i in range(len(FSamples)):
        tidx[idx:idx+FSamples[i]] = i
        idx += FSamples[i]
    N = np.zeros((npoints, 3)) #Allocate space for normals
    idx = 0

    #Vector used to determine if points need to be flipped across parallelogram
    V3 = P2 - P1
    V3 = V3/np.sqrt(np.sum(V3**2, 1))[:, None] #Normalize

    #Randomly sample points on each face
    #Generate random points uniformly in parallelogram
    u = np.random.rand(npoints, 1)
    v = np.random.rand(npoints, 1)
    Ps = u*V1[tidx, :] + P0[tidx, :]
    Ps += v*V2[tidx, :]
    #Flip over points which are on the other side of the triangle
    dP = Ps - P1[tidx, :]
    proj = np.sum(dP*V3[tidx, :], 1)
    dPPar = V3[tidx, :]*proj[:, None] #Parallel project onto edge
    dPPerp = dP - dPPar
    Qs = Ps - dPPerp
    dP0QSqr = np.sum((Qs - P0[tidx, :])**2, 1)
    dP0PSqr = np.sum((Ps - P0[tidx, :])**2, 1)
    idxreg = np.arange(npoints, dtype=np.int64)
    idxflip = idxreg[dP0QSqr < dP0PSqr]
    u[idxflip, :] = 1 - u[idxflip, :]
    v[idxflip, :] = 1 - v[idxflip, :]
    Ps[idxflip, :] = P0[tidx[idxflip], :] + u[idxflip, :]*V1[tidx[idxflip], :] + v[idxflip, :]*V2[tidx[idxflip], :]

    #Step 3: Compute normals of sampled points by barycentric interpolation
    Ns = u*VNormals[ITris[tidx, 1], :]
    Ns += v*VNormals[ITris[tidx, 2], :]
    Ns += (1-u-v)*VNormals[ITris[tidx, 0], :]

    if colPoints:
        return (Ps.T, Ns.T)
    return (Ps, Ns)



def get_edges(VPos, ITris):
    """
    Given a list of triangles, return an array representing the edges
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    Returns: I, J
        Two parallel 1D arrays with indices of edges
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return L.nonzero()
