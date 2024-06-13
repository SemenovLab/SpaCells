from copy import deepcopy
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import Delaunay
from collections import defaultdict


def getAlphaShapes(points, alpha, debug=False):
    """
    Compute the alpha shape of a set of points.

    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :returns: set of (i,j) point pairs representing edges of the alpha-shape.
    """
    assert points.shape[0] > 3, "Need at least four points"

    # triangulate all points
    tri = Delaunay(points)

    pa = points[tri.simplices[:, 0]]
    pb = points[tri.simplices[:, 1]]
    pc = points[tri.simplices[:, 2]]
    a = np.sqrt((pa[:, 0] - pb[:, 0]) ** 2 + (pa[:, 1] - pb[:, 1]) ** 2)
    b = np.sqrt((pb[:, 0] - pc[:, 0]) ** 2 + (pb[:, 1] - pc[:, 1]) ** 2)
    c = np.sqrt((pc[:, 0] - pa[:, 0]) ** 2 + (pc[:, 1] - pa[:, 1]) ** 2)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    # Computing radius of triangle circumcircle
    # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
    circum_r = a * b * c / (4.0 * area + 1e-10)
    verts = tri.simplices[circum_r < alpha]

    # get edges that only appear once
    edges = _getUniqueEdges(
        np.concatenate([verts[:, [0, 1]], verts[:, [1, 2]], verts[:, [0, 2]]], axis=0)
    )
    # order all edges
    edge_component_indices = _getOrderedEdgeComponents(edges)
    if debug:
        print("edge_component_indices:", len(edge_component_indices))

    # points
    edge_components = []
    for component in edge_component_indices:
        edge_components.append(points[component])
        if debug:
            print(component.shape)
    return edge_components


def _bfsGetShortestRing(edge_dict, start_point):
    """
    Given a set of edges, return the shortest ring starting from start_point.
    Helper function for `_getOrderedEdgeComponents`

    :param edge_dict: a dict of edges. key is a point; value = set(its neighbors)
    :param start_point: the starting point of the ring
    :returns: a list of points representing the shortest ring
    """
    bfs_queue = [(start_point, [start_point])]
    while len(bfs_queue) > 0:
        cur_point, path = bfs_queue.pop(0)
        if cur_point == start_point and len(path) > 2:  # found ring
            return path[:-1]
        for neighbor in edge_dict[cur_point]:
            if neighbor not in path or (neighbor == start_point and len(path) > 2):
                bfs_queue.append((neighbor, path + [neighbor]))
    raise Exception("Open ring found")


def _getOrderedEdgeComponents(edges):
    """
    Given a set of edges, return a list of ordered edge components
    Helper function for `getAlphaShapes`

    :param edges: np.array of shape (n,2) for n edges.
    :returns: a list of ordered edge components. Each component 
        is a np.array of shape (m,2) for m edges.
    """

    # key is a point; value = set(its neighbors)
    edge_dict = defaultdict(set)
    for i in range(edges.shape[0]):
        edge_dict[edges[i, 0]].add(edges[i, 1])
        edge_dict[edges[i, 1]].add(edges[i, 0])
    edge_dict, components = _pruneTouchingComponents(edge_dict)
    cur_point = next(iter(edge_dict))
    ordered_edges = []
    not_visited = set(edge_dict.keys())
    while len(not_visited) > 0:
        next_point = None
        for point in edge_dict[cur_point]:
            if point in not_visited:
                edge_dict[cur_point].remove(point)
                edge_dict[point].remove(cur_point)
                next_point = point
                ordered_edges.append(next_point)
                if len(edge_dict[cur_point]) == 0:
                    not_visited.remove(cur_point)
                break
        if next_point is None:
            not_visited.discard(cur_point)
            components.append(np.array(ordered_edges))
            ordered_edges = []
            if len(not_visited) == 0:
                break
            next_point = next(iter(not_visited))
        cur_point = next_point

    edge_components = []
    for component in components:
        component = np.stack([np.roll(component, 1), component], axis=1)
        edge_components.append(component)
    return edge_components


def _getUniqueEdges(all_edges):
    """
    Return the boundary of Delaunay represented by all_edges.
    Helper function for `getAlphaShapes`

    :param all_edges: np.array of shape (n,2) for n edges.
    :returns: np.array of shape (m,2) for m edges, where m <= n.
    """
    all_edges = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def _pruneTouchingComponents(edge_dict):
    """
    Given ring components, separate touching rings by extracting the smallest rings.
    Helper function for `_getOrderedEdgeComponents`

    :param edge_dict: a dict of edges. key is a point; value = set(its neighbors)
    :returns: tuple of (edge_dict with touching components pruned, list of touching components)
    """
    edge_dict = deepcopy(edge_dict)
    components = []
    touching_points = set()
    for point in edge_dict:
        if len(edge_dict[point]) > 2:
            touching_points.add(point)
    while len(touching_points) > 0:
        cur_point = next(iter(touching_points))
        component = _bfsGetShortestRing(edge_dict, cur_point)
        for i in range(len(component)):
            j = (i + 1) % len(component)
            edge_dict[component[i]].remove(component[j])
            edge_dict[component[j]].remove(component[i])
        # separate loop to ensure all components are checked AFTER updating edge_dict
        for i in range(len(component)):
            if component[i] in touching_points and len(edge_dict[component[i]]) <= 2:
                touching_points.remove(component[i])
        components.append(np.array(component))
    new_edge_dict = {}
    for point in edge_dict:
        if len(edge_dict[point]) > 0:
            new_edge_dict[point] = edge_dict[point]
    return new_edge_dict, components


def getComponents(boundary, keep_holes=True):
    """
    Get the components of a boundary defined by a MultiPolygon.

    :param boundary: the boundary to get components from
    :param keep_holes: whether to keep holes
    :returns: a list of components, where each component is a MultiPolygon
    """
    components = []
    for geom in boundary.geoms:
        if keep_holes:
            if geom.geom_type == "Polygon":
                geom = MultiPolygon([geom])
            components.append(geom)
        else:
            poly = Polygon(geom.exterior.coords)
            components.append(MultiPolygon([poly]))
    return components


def getHoles(boundary):
    """
    Get the holes within boundary components.

    :param boundary: the boundary to get holes from
    :returns: a list of holes, where each hole is a MultiPolygon
    """
    external_components = getComponents(boundary, keep_holes=False)
    external_boundary = shapely.unary_union(external_components)
    holes_multi = external_boundary - boundary
    if holes_multi.geom_type == "Polygon":
        holes_multi = MultiPolygon([holes_multi])
    holes = getComponents(holes_multi, keep_holes=True)
    return holes


def pruneSmallComponents(
    boundary, min_area=0, min_edges=0, holes_min_area=0, holes_min_edges=0
):
    """
    Prune small components from a boundary defined by a MultiPolygon.

    :param boundary: the boundary to prune
    :param min_area: the minimum area of a polygon component to keep
    :param min_edges: the minimum number of edges of a polygon component to keep
    :param holes_min_area: the minimum area of a hole to keep the hole
    :param holes_min_edges: the minimum number of edges of a hole to keep the hole
    :returns: the pruned boundary
    """
    polygons = []
    for geom in boundary.geoms:
        holes_to_keep = []
        for hole in geom.interiors:
            if (
                Polygon(hole).area >= holes_min_area
                and len(hole.coords) - 1 >= holes_min_edges
            ):
                holes_to_keep.append(hole)
        if geom.area >= min_area and len(geom.exterior.coords) - 1 >= min_edges:
            polygons.append(Polygon(geom.exterior.coords, holes_to_keep))
    return MultiPolygon(polygons)