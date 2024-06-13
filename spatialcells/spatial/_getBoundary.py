import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

from ._utils import *


def getBoundary(anndata, communitycolumn, communityIndexList, alpha=100, debug=False):
    """
    Get a boundary for the communities of interest as a Shapely MultiPolygon.

    The boundary is defined based on the alpha shape of points in the
    communities. The alpha shape is the generalization of the convex hull,
    and is generated via Delaunay triangulation of all the points of interest.
    The alpha parameter controls the longest edge that can appear in the alpha
    shape. Smaller alpha gives more detailed boundary, but may appear more
    jagged and may leave out some points that are too far away from the rest
    of the points.

    :param anndata: the anndata object
    :param communitycolumn: the column name of the community
    :param communityIndexList: the list of community indexes
    :param alpha: the alpha parameter for alpha shape. Smaller alpha gives
        more detailed boundary, but may appear more jagged.
    :param debug: whether to return the polygons and edge components
    :returns: the boundaries of components as a MultiPolygon object
    """

    xy = anndata.obs[anndata.obs[communitycolumn].isin(communityIndexList)][
        ["X_centroid", "Y_centroid"]
    ].to_numpy()
    edge_components = getAlphaShapes(xy, alpha)
    polygons = [Polygon(edge[:, 0, :]).buffer(0) for edge in edge_components]
    boundary = MultiPolygon(polygons)
    # make sure the boundary is valid and points of interest are inside
    boundary = make_valid(boundary).buffer(alpha / 100)
    if boundary.geom_type == "Polygon":
        boundary = MultiPolygon([boundary])

    # Ensure that holes do not intersect each other or the boundary
    new_boundary = []
    for poly in boundary.geoms:
        holes = poly.interiors
        hole_polygons = [Polygon(hole) for hole in holes]
        # set limit to shrink holes tangent to the boundary
        hole_limit = Polygon(poly.exterior).buffer(-alpha / 100)
        hole_multipoly = shapely.unary_union(hole_polygons) & hole_limit
        if hole_multipoly.geom_type == "Polygon":
            hole_multipoly = MultiPolygon([hole_multipoly])
        new_holes_polygons = [p.exterior.coords for p in hole_multipoly.geoms]
        new_poly = Polygon(poly.exterior, new_holes_polygons)
        new_boundary.append(new_poly)
    boundary = make_valid(MultiPolygon(new_boundary))

    if debug:
        return boundary, polygons, edge_components

    return boundary
