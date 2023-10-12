from astropy.io import ascii, fits
import numpy as np
from astropy.table import Table, vstack, hstack, Column
from shapely.geometry.polygon import LinearRing
import scipy.spatial
import sys

def do_kdtree(combined_x_y_arrays, points, k=1):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    # dist, indexes = mytree.query(points, k)
    return mytree.query(points, k=k)

def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result

def intersections(a, b):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)

    x = [p.x for p in mp.geoms]
    y = [p.y for p in mp.geoms]
    return x, y

path = sys.argv[1]
scenes = int(sys.argv[2])
shears = int(sys.argv[3])
method = sys.argv[4]

catalog = ascii.read(path + "/shear_catalog.dat")

for i in range(scenes):
    for j in range(shears):
        if method == "LF":
            cancel = 4
        else:
            cancel = 1
        for k in range(cancel):
            if method == "LF":
                catalog_cut = catalog[(catalog["scene_index"] == i) & (catalog["shear_index"] == j) & (catalog["cancel_index"] == k)]
            else:
                catalog_cut = catalog[
                    (catalog["scene_index"] == i) & (catalog["shear_index"] == j)]

            blends = []
            positions = np.vstack([catalog_cut["position_x"], catalog_cut["position_y"]]).T
            dist, ind = do_kdtree(positions, positions, k=[2])
            for l in range(len(catalog_cut)):
                ellipses = [(catalog_cut["position_x"][l], catalog_cut["position_y"][l],
                             catalog_cut["a_image"][l] * catalog_cut["kron_radius"][l],
                             catalog_cut["b_image"][l] * catalog_cut["kron_radius"][l], catalog_cut["elongation"][l]),
                            (catalog_cut["position_x"][ind[l]], catalog_cut["position_y"][ind[l]],
                             catalog_cut["a_image"][ind[l]] * catalog_cut["kron_radius"][ind[l]],
                             catalog_cut["b_image"][ind[l]] * catalog_cut["kron_radius"][ind[l]],
                             catalog_cut["elongation"][ind[l]])]

                try:
                    a, b = ellipse_polyline(ellipses)
                    x, y = intersections(a, b)
                    blends.append(1)

                except AttributeError:
                    connection_vector = np.array([catalog_cut["position_x"][l] - catalog_cut["position_x"][ind[l]],
                                                  catalog_cut["position_y"][l] - catalog_cut["position_y"][ind[l]]]).reshape(2)

                    ellip1a = np.array([catalog_cut["a_image"][l] * catalog_cut["kron_radius"][l]
                                        * np.cos(catalog_cut["elongation"][l] * np.pi / 180),
                                        catalog_cut["a_image"][l] * catalog_cut["kron_radius"][l]
                                        * np.sin(catalog_cut["elongation"][l] * np.pi / 180)])

                    ellip2a = np.array([catalog_cut["a_image"][ind[l]] * catalog_cut["kron_radius"][ind[l]]
                                        * np.cos(catalog_cut["elongation"][ind[l]] * np.pi / 180),
                                        catalog_cut["a_image"][ind[l]] * catalog_cut["kron_radius"][ind[l]]
                                        * np.sin(catalog_cut["elongation"][ind[l]] * np.pi / 180)])
                    v_norm = np.sqrt(sum(connection_vector ** 2))

                    proj_of_u_on_v = (np.dot(connection_vector, ellip1a) / v_norm ** 2) * connection_vector

                    norm = np.sqrt(sum(proj_of_u_on_v ** 2))

                    proj_of_u_on_v2 = (np.dot(connection_vector, ellip2a) / v_norm ** 2) * connection_vector

                    norm2 = np.sqrt(sum(proj_of_u_on_v2 ** 2))

                    if (norm >= v_norm) or (norm2 >= v_norm):
                        blends.append(1)
                    else:
                        blends.append(0)


            col_c = Column(name='kron_blend', data=blends)

            catalog_cut.add_column(col_c)

            if (i == 0) and (j == 0) and (k == 0):
                catalog_out = catalog_cut
            else:
                catalog_out = vstack([catalog_out, catalog_cut])

ascii.write(catalog_out, path + "/shear_catalog.dat", overwrite=True)
