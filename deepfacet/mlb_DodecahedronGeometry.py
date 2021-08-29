from molecular_builder.geometry import PlaneGeometry
import numpy as np

class DodecahedronGeometry(PlaneGeometry):
    """
    Taken from molecular builder source code: https://github.com/henriasv/molecular-builder/

    A convex rectangular dodecahedron geometry to be used for silicon
    carbide (SiC).
    :param d: (shortest) length from dodecahedron center to sides
    :type d: float
    :param center: center of dodecahedron
    :type center: array_like
    """
    def __init__(self, d, center=[0, 0, 0]):
        # make list of normal vectors
        lst = [[+1, +1, 0], [+1, 0, +1], [0, +1, +1], [+1, -1, 0],
               [+1, 0, -1], [+0, 1, -1], [-1, +1, 0], [-1, 0, +1],
               [0, -1, +1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1]]
        normals = np.asarray(lst, dtype=int)
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # find points in planes
        points = d * normals + np.asarray(center)
        super().__init__(points, normals)
