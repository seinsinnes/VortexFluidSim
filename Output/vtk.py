import numpy as np
from evtk.hl import pointsToVTK, gridToVTK




def write(filename, points):
    pointsX = []
    pointsY = []
    pointsZ = []
    pointsU = []
    pointsV = []
    pointsW = []
    for pt in points:
        pointsX.append(pt.position[0])
        pointsY.append(pt.position[1])
        pointsZ.append(pt.position[2])
        pointsU.append(pt.velocity[0])
        pointsV.append(pt.velocity[1])
        pointsW.append(pt.velocity[2])
    pointsToVTK(filename, np.array(pointsX), np.array(pointsY), np.array(pointsZ),
                data={"u": np.array(pointsU), "v": np.array(pointsV), "w": np.array(pointsW)})

def writeGrid(filename,grid):

    # Dimensions

    nx, ny, nz = grid.getNumCells( 0 ), grid.getNumCells( 1 ), grid.getNumCells( 2 )

    lx, ly, lz = grid.getExtent()

    dx, dy, dz = lx / nx, ly / ny, lz / nz

    ncells = nx * ny * nz

    npoints = (nx + 1) * (ny + 1) * (nz + 1)

    # Coordinates

    x = np.arange(0, lx + 0.1 * dx, dx, dtype='float64')

    y = np.arange(0, ly + 0.1 * dy, dy, dtype='float64')

    z = np.arange(0, lz + 0.1 * dz, dz, dtype='float64')

    gridContArray = np.array(grid.contents)


    gridContArrayU = np.ascontiguousarray(gridContArray[:, :, :, 0])
    gridContArrayV = np.ascontiguousarray(gridContArray[:, :, :, 1])
    gridContArrayW = np.ascontiguousarray(gridContArray[:, :, :, 2])


    gridToVTK(filename, x, y, z, pointData={"u": gridContArrayU, "v": gridContArrayV, "w": gridContArrayW})