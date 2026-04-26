import math

def point_in_polygon(point, poly_vertices):
    """射线法判断点是否在多边形内部"""
    x, y = point
    inside = False
    n = len(poly_vertices)
    for i in range(n):
        x1, y1 = poly_vertices[i]
        x2, y2 = poly_vertices[(i+1)%n]
        # 检查点是否在边界上
        if point_on_segment(point, (x1, y1), (x2, y2)):
            return True
        # 射线法
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1)+x1):
            inside = not inside
    return inside

def closest_point_on_polygon_boundary(point, poly_vertices):
    """找到多边形边界上离给定点最近的点（投影到每条边上）"""
    px, py = point
    best_dist = float('inf')
    best_point = None
    n = len(poly_vertices)
    for i in range(n):
        x1, y1 = poly_vertices[i]
        x2, y2 = poly_vertices[(i+1)%n]
        ax, ay = x2-x1, y2-y1
        t = ((px-x1)*ax + (py-y1)*ay) / (ax*ax+ay*ay+1e-12)
        t = max(0, min(1, t))
        proj_x = x1 + t*ax
        proj_y = y1 + t*ay
        dist = (proj_x-px)**2 + (proj_y-py)**2
        if dist < best_dist:
            best_dist = dist
            best_point = (proj_x, proj_y)
    return best_point

def detour_around_obstacle_robust(A, B, obs, safety_meters):
    """
    生成绕行单个障碍物的路径（确保不与障碍物相交）
    支持任意大小（极小到极大）的凸多边形/简单多边形
    """
    # 如果起点和终点都在障碍物内部，这种情况无法不穿越障碍物连接，
    # 但我们可以先走到边界，沿边界走，再走到另一个边界点。
    A_inside = point_in_polygon(A, obs["vertices"])
    B_inside = point_in_polygon(B, obs["vertices"])

    # 若某端点在内部，将其替换为边界上的最近点
    A_eff = A
    B_eff = B
    if A_inside:
        A_eff = closest_point_on_polygon_boundary(A, obs["vertices"])
    if B_inside:
        B_eff = closest_point_on_polygon_boundary(B, obs["vertices"])

    # 计算扩展矩形（安全边界）
    minx, miny, maxx, maxy = get_bounding_box(obs["vertices"])
    expand_deg = safety_meters / 111000.0 + 1e-6
    minx -= expand_deg
    miny -= expand_deg
    maxx += expand_deg
    maxy += expand_deg
    rect_pts = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    edges = [(rect_pts[0], rect_pts[1]), (rect_pts[1], rect_pts[2]),
             (rect_pts[2], rect_pts[3]), (rect_pts[3], rect_pts[0])]

    # 方法1：矩形边界最短路径（折线）
    intersections = []
    for e in edges:
        inter = line_intersection_point(A_eff, B_eff, e[0], e[1])
        if inter:
            if not any(math.hypot(inter[0]-p[0], inter[1]-p[1]) < 1e-9 for p in intersections):
                intersections.append(inter)
    # 如果交点不足2个，说明线段完全在矩形内部或者完全在外无交点
    # 完全在内部 -> 需要沿矩形边界走
    # 完全在外无交点 -> 不需要绕行
    if len(intersections) >= 2:
        intersections.sort(key=lambda p: math.hypot(p[0]-A_eff[0], p[1]-A_eff[1]))
        p_enter, p_exit = intersections[0], intersections[1]

        def edge_index(pt):
            for i, e in enumerate(edges):
                if point_on_segment(pt, e[0], e[1]):
                    return i
            return -1

        enter_idx = edge_index(p_enter)
        exit_idx = edge_index(p_exit)

        if enter_idx != -1 and exit_idx != -1:
            # 构建沿矩形边界的顺时针/逆时针路径
            def build_path(start_pt, start_idx, end_pt, end_idx, direction=1):
                path = [start_pt]
                idx = start_idx
                while True:
                    next_idx = (idx + direction) % 4
                    if direction == 1:
                        next_pt = edges[idx][1]
                    else:
                        next_pt = edges[idx][0]
                    if idx == end_idx:
                        path.append(end_pt)
                        break
                    else:
                        path.append(next_pt)
                    idx = next_idx
                    if idx == start_idx:
                        break
                return path

            path_cw = build_path(p_enter, enter_idx, p_exit, exit_idx, 1)
            path_ccw = build_path(p_enter, enter_idx, p_exit, exit_idx, -1)

            def total_len(path):
                return (math.hypot(path[0][0]-A_eff[0], path[0][1]-A_eff[1]) +
                        sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1)) +
                        math.hypot(path[-1][0]-B_eff[0], path[-1][1]-B_eff[1]))

            polyline = [A_eff] + (path_cw if total_len(path_cw) <= total_len(path_ccw) else path_ccw) + [B_eff]
            if not path_intersects_any_obstacle(polyline, [obs]):
                smooth = catmull_rom_spline(polyline, num_segments=25)
                if not path_intersects_any_obstacle(smooth, [obs]):
                    return smooth
                else:
                    return polyline

    # 方法2：垂直偏移法（尝试多个偏移方向和幅度）
    dx = B_eff[0] - A_eff[0]
    dy = B_eff[1] - A_eff[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return [A, B]

    dx /= length
    dy /= length
    perp_x = -dy
    perp_y = dx
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # 动态偏移量：从安全距离开始，逐步增加
    for mult in [1.0, 2.0, 4.0, 8.0, 16.0]:
        offset = expand_deg * mult
        left = (center_x + perp_x * offset, center_y + perp_y * offset)
        right = (center_x - perp_x * offset, center_y - perp_y * offset)
        for p in [left, right]:
            # 确保两段都不与障碍物相交
            if (not polygon_intersects_segment(obs["vertices"], A_eff, p) and
                not polygon_intersects_segment(obs["vertices"], p, B_eff)):
                # 可选：再检查点p是否在障碍物内部
                if not point_in_polygon(p, obs["vertices"]):
                    return [A_eff, p, B_eff]

    # 方法3：最后保底：沿多边形边界行走（寻找两个最近边界点）
    # 找到A_eff到多边形最近点，B_eff到多边形最近点，然后沿边界连接
    nearest_A = closest_point_on_polygon_boundary(A_eff, obs["vertices"])
    nearest_B = closest_point_on_polygon_boundary(B_eff, obs["vertices"])

    # 沿多边形边界走两种方向，取较短路径
    poly_vertices = obs["vertices"]
    n = len(poly_vertices)
    # 找到nearest_A和nearest_B分别在哪个边上以及参数位置（简化处理：直接取最近顶点并沿顶点序列走）
    # 更精确：先找到最近边的索引，然后沿顶点环前进。为简化且仍可靠，我们取最近顶点
    def nearest_vertex_index(point, vertices):
        min_idx = 0
        min_dist = math.hypot(point[0]-vertices[0][0], point[1]-vertices[0][1])
        for i, v in enumerate(vertices):
            d = math.hypot(point[0]-v[0], point[1]-v[1])
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx

    idxA = nearest_vertex_index(nearest_A, poly_vertices)
    idxB = nearest_vertex_index(nearest_B, poly_vertices)

    # 构建沿多边形环的顺时针路径（取顶点序列）
    def boundary_path(start_idx, end_idx, vertices, forward=True):
        path = [vertices[start_idx]]
        idx = start_idx
        while idx != end_idx:
            if forward:
                idx = (idx + 1) % n
            else:
                idx = (idx - 1 + n) % n
            path.append(vertices[idx])
        return path

    path_fwd = boundary_path(idxA, idxB, poly_vertices, forward=True)
    path_rev = boundary_path(idxA, idxB, poly_vertices, forward=False)
    boundary_route = path_fwd if len(path_fwd) <= len(path_rev) else path_rev

    # 将起点终点替换为原始A,B（如果原来在内部，已经替换过）
    final_route = [A_eff] + boundary_route + [B_eff]
    # 去重合并相邻重复点
    unique_route = []
    for p in final_route:
        if not unique_route or math.hypot(p[0]-unique_route[-1][0], p[1]-unique_route[-1][1]) > 1e-9:
            unique_route.append(p)
    if not path_intersects_any_obstacle(unique_route, [obs]):
        return unique_route

    # 实在不行，返回原始两点（但几乎不会触发）
    return [A, B]
