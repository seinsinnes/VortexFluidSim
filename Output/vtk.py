import numpy as np
from evtk.hl import pointsToVTK




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