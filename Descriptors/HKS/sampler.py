import argparse
from trimesh import load_off, sample_by_area

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to OFF file for triangle mesh on which to compute the HKS")
    parser.add_argument("--output", type=str, required=True, help="Path to text file which will holds the sampled points and their normals")
    parser.add_argument("--npoints", type=float, required=True, help="Number of points to sample")
    parser.add_argument("--do_plot", type=int, default=0, help="Whether to plot the result with matplotlib")
    opt = parser.parse_args()
    (VPos, VColors, ITris) = load_off(opt.input)
    npoints = int(opt.npoints)
    Ps, Ns = sample_by_area(VPos, ITris, npoints, colPoints=False)
    if opt.do_plot == 1:
        import numpy as np 
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Ps[:, 0], Ps[:, 1], Ps[:, 2])
        plt.show()
    X = np.concatenate((Ps, Ns), axis=1)
    fout = open(opt.output, "w")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fout.write("{}".format(X[i, j]))
            if j < X.shape[1]-1:
                fout.write(",")
        fout.write("\n")
    fout.close()