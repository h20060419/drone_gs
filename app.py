import streamlit as st
import pandas as pd
import time
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from heartbeat_sim import HeartbeatSimulator
import math
import json
import os

# ========== 障碍物持久化 ==========
OBSTACLE_FILE = "obstacles.json"

def validate_obstacles(obstacles):
    valid = []
    for obs in obstacles:
        if not isinstance(obs, dict):
            continue
        vertices = obs.get("vertices", [])
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        valid_vertices = []
        for v in vertices:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lng, lat = v[0], v[1]
                if isinstance(lng, (int, float)) and isinstance(lat, (int, float)):
                    valid_vertices.append((float(lng), float(lat)))
        if len(valid_vertices) >= 3:
            height = obs.get("height", 30.0)
            if isinstance(height, (int, float)):
                valid.append({"vertices": valid_vertices, "height": float(height)})
    return valid

def save_obstacles_to_file(obstacles):
    try:
        with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(obstacles, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def load_obstacles_from_file():
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return validate_obstacles(data)
                else:
                    return []
        except Exception as e:
            print(f"加载失败: {e}")
            return []
    return []

# ========== GCJ-02 转 WGS-84 ==========
def gcj02_to_wgs84(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    PI = math.pi
    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * PI) + 40.0 * math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 * math.sin(lng * PI / 30.0)) * 2.0 / 3.0
        return ret
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * PI)
    wgs_lat = lat - dlat
    wgs_lng = lng - dlng
    return wgs_lng, wgs_lat

# ========== 几何辅助函数 ==========
def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def cross(ax, ay, bx, by):
        return ax*by - ay*bx
    def on_segment(px, py, qx, qy, rx, ry):
        return min(px, qx) <= rx <= max(px, qx) and min(py, qy) <= ry <= max(py, qy)
    o1 = cross(x2-x1, y2-y1, x3-x1, y3-y1)
    o2 = cross(x2-x1, y2-y1, x4-x1, y4-y1)
    o3 = cross(x4-x3, y4-y3, x1-x3, y1-y3)
    o4 = cross(x4-x3, y4-y3, x2-x3, y2-y3)
    if o1 == 0 and on_segment(x1, y1, x2, y2, x3, y3): return True
    if o2 == 0 and on_segment(x1, y1, x2, y2, x4, y4): return True
    if o3 == 0 and on_segment(x3, y3, x4, y4, x1, y1): return True
    if o4 == 0 and on_segment(x3, y3, x4, y4, x2, y2): return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def point_on_segment(p, a, b):
    x, y = p; x1, y1 = a; x2, y2 = b
    cross = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
    if abs(cross) > 1e-9:
        return False
    dot = (x - x1)*(x2 - x1) + (y - y1)*(y2 - y1)
    if dot < 0 or dot > (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1):
        return False
    return True

def polygon_intersects_segment(poly_vertices, seg_start, seg_end):
    try:
        n = len(poly_vertices)
        if n < 3:
            return False
        for i in range(n):
            x1, y1 = poly_vertices[i]
            x2, y2 = poly_vertices[(i+1)%n]
            if segments_intersect(seg_start[0], seg_start[1], seg_end[0], seg_end[1], x1, y1, x2, y2):
                return True
        mid_x = (seg_start[0] + seg_end[0]) / 2
        mid_y = (seg_start[1] + seg_end[1]) / 2
        inside = False
        for i in range(n):
            x1, y1 = poly_vertices[i]
            x2, y2 = poly_vertices[(i+1)%n]
            if ((y1 > mid_y) != (y2 > mid_y)) and (mid_x < (x2 - x1) * (mid_y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    except:
        return False

def get_bounding_box(poly_vertices):
    xs = [v[0] for v in poly_vertices]
    ys = [v[1] for v in poly_vertices]
    return min(xs), min(ys), max(xs), max(ys)

def line_intersection_point(p1, p2, p3, p4):
    try:
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-12:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return (x, y)
        return None
    except:
        return None

def path_intersects_any_obstacle(path_points, obstacles, flight_height=None):
    if not obstacles or len(path_points) < 2:
        return False
    for i in range(len(path_points)-1):
        seg_start = path_points[i]
        seg_end = path_points[i+1]
        for obs in obstacles:
            if flight_height is not None and flight_height >= obs["height"]:
                continue
            if polygon_intersects_segment(obs["vertices"], seg_start, seg_end):
                return True
    return False

# ========== 平滑曲线 ==========
def catmull_rom_spline(points, num_segments=20):
    if not points or len(points) < 2:
        return points
    if len(points) == 2:
        return [points[0] + (points[1]-points[0]) * t for t in [i/num_segments for i in range(num_segments+1)]]
    result = []
    for i in range(len(points)-1):
        p0 = points[max(i-1, 0)]
        p1 = points[i]
        p2 = points[i+1]
        p3 = points[min(i+2, len(points)-1)]
        for t in [j/num_segments for j in range(num_segments)]:
            t2 = t*t
            t3 = t2*t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            result.append((x, y))
    result.append(points[-1])
    return result

# ========== 增强版绕行（保证不与当前障碍物相交）==========
def detour_around_obstacle_robust(A, B, obs, safety_meters):
    """生成绕行单个障碍物的路径，确保不与障碍物相交，如果平滑曲线失败则回退到折线，再失败则使用垂直偏移法"""
    # 计算扩展矩形
    minx, miny, maxx, maxy = get_bounding_box(obs["vertices"])
    expand_deg = safety_meters / 111000.0 + 1e-6
    minx -= expand_deg
    miny -= expand_deg
    maxx += expand_deg
    maxy += expand_deg
    rect_pts = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]

    # 方法1：矩形边界最短路径（折线）
    edges = [(rect_pts[0], rect_pts[1]), (rect_pts[1], rect_pts[2]), (rect_pts[2], rect_pts[3]), (rect_pts[3], rect_pts[0])]
    intersections = []
    for e in edges:
        inter = line_intersection_point(A, B, e[0], e[1])
        if inter:
            if not any(math.hypot(inter[0]-p[0], inter[1]-p[1]) < 1e-9 for p in intersections):
                intersections.append(inter)
    if len(intersections) >= 2:
        intersections.sort(key=lambda p: math.hypot(p[0]-A[0], p[1]-A[1]))
        p_enter, p_exit = intersections[0], intersections[1]
        def edge_index(pt):
            for i, e in enumerate(edges):
                if point_on_segment(pt, e[0], e[1]):
                    return i
            return -1
        enter_idx = edge_index(p_enter)
        exit_idx = edge_index(p_exit)
        if enter_idx != -1 and exit_idx != -1:
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
                return math.hypot(path[0][0]-A[0], path[0][1]-A[1]) + sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1)) + math.hypot(path[-1][0]-B[0], path[-1][1]-B[1])
            polyline = [A] + (path_cw if total_len(path_cw) <= total_len(path_ccw) else path_ccw) + [B]
            # 检查折线是否与障碍物相交
            if not path_intersects_any_obstacle(polyline, [obs]):
                # 尝试平滑化
                smooth = catmull_rom_spline(polyline, num_segments=25)
                if not path_intersects_any_obstacle(smooth, [obs]):
                    return smooth
                else:
                    return polyline
    # 方法2：垂直偏移法（强制生成外侧点）
    # 计算AB方向及垂直方向
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return [A, B]
    dx /= length
    dy /= length
    perp_x = -dy
    perp_y = dx
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    # 逐步增加偏移距离
    for mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
        offset = expand_deg * mult
        left = (center_x + perp_x * offset, center_y + perp_y * offset)
        right = (center_x - perp_x * offset, center_y - perp_y * offset)
        # 检查左右两侧路径
        for p in [left, right]:
            if not polygon_intersects_segment(obs["vertices"], A, p) and not polygon_intersects_segment(obs["vertices"], p, B):
                return [A, p, B]
    # 最后的回退：直接返回原直线（理论上不会执行到这里）
    return [A, B]

def generate_detour_route(A, B, obstacles, flight_height, safety_meters, max_attempts=3):
    """
    多障碍物绕行，如果失败则自动增加安全距离重试
    """
    original_safety = safety_meters
    for attempt in range(max_attempts):
        current_safety = safety_meters * (1 + attempt * 0.5)  # 递增50%
        current_route = [A, B]
        for _ in range(10):  # 内层迭代次数
            if not path_intersects_any_obstacle(current_route, obstacles, flight_height):
                return current_route
            new_route = [current_route[0]]
            for i in range(len(current_route)-1):
                seg_start = current_route[i]
                seg_end = current_route[i+1]
                target_obs = None
                for obs in obstacles:
                    if flight_height < obs["height"] and polygon_intersects_segment(obs["vertices"], seg_start, seg_end):
                        target_obs = obs
                        break
                if target_obs is None:
                    new_route.append(seg_end)
                else:
                    detour_seg = detour_around_obstacle_robust(seg_start, seg_end, target_obs, current_safety)
                    new_route.extend(detour_seg[1:])
            current_route = new_route
        # 如果内层循环结束仍有冲突，增加安全距离重试
        if not path_intersects_any_obstacle(current_route, obstacles, flight_height):
            return current_route
    # 最终失败，返回原始直线（但会给出警告）
    return [A, B]

# ========== Streamlit 页面配置 ==========
st.set_page_config(page_title="无人机地面站监控系统", layout="wide")

if "app_version" not in st.session_state or st.session_state.app_version != "v20_final_fixed":
    st.session_state.sim = HeartbeatSimulator()
    st.session_state.history = []
    loaded = load_obstacles_from_file()
    st.session_state.obstacles = loaded if loaded else []
    st.session_state.default_obstacle_height = 30.0
    st.session_state.safety_distance = 3.0
    st.session_state.detour_route = None
    st.session_state.app_version = "v20_final_fixed"
else:
    if st.session_state.obstacles:
        st.session_state.obstacles = validate_obstacles(st.session_state.obstacles)

st.sidebar.title("🧭 导航控制")
page = st.sidebar.radio("请选择功能页面", ["航线规划", "飞行监控"])
st.sidebar.divider()
coord_mode = st.sidebar.radio("坐标系设置（输入坐标的原始类型）", ["WGS-84", "GCJ-02"], index=0)
st.sidebar.info("✅ 卫星图底图：Esri World Imagery (WGS-84)\n若选择 GCJ-02，系统会自动转换为 WGS-84 匹配卫星图。")

if page == "航线规划":
    st.header("🗺️ 航线规划 + 障碍物圈选 (多障碍物可靠绕行)")

    st.sidebar.subheader("🚧 障碍物默认高度")
    default_h = st.sidebar.number_input(
        "新绘制障碍物的默认高度 (米)", 
        min_value=0.0, max_value=200.0, 
        value=st.session_state.default_obstacle_height, step=5.0
    )
    st.session_state.default_obstacle_height = default_h
    st.sidebar.divider()

    st.sidebar.subheader("🛡️ 基础安全距离 (米)")
    safety = st.sidebar.number_input(
        "绕行安全距离", 
        min_value=0.0, max_value=200.0, 
        value=st.session_state.safety_distance, step=5.0,
        help="若无法避障，会自动增加安全距离重试"
    )
    st.session_state.safety_distance = safety
    st.sidebar.divider()

    st.sidebar.subheader("📋 已添加的障碍物")
    if not st.session_state.obstacles:
        st.sidebar.write("暂无障碍物")
    else:
        for idx, obs in enumerate(st.session_state.obstacles):
            with st.sidebar.expander(f"障碍物 {idx+1} (高度: {obs['height']} m)"):
                new_height = st.number_input(
                    f"高度 (m)", min_value=0.0, max_value=200.0, value=obs['height'],
                    key=f"height_{idx}", step=5.0
                )
                if new_height != obs['height']:
                    obs['height'] = new_height
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.rerun()
                if st.button(f"🗑️ 删除障碍物 {idx+1}", key=f"del_{idx}"):
                    st.session_state.obstacles.pop(idx)
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.session_state.detour_route = None
                    st.rerun()
                st.caption(f"顶点数: {len(obs['vertices'])}")

    st.sidebar.metric("障碍物总数", len(st.session_state.obstacles))
    st.sidebar.divider()
    col_save1, col_save2 = st.sidebar.columns(2)
    with col_save1:
        if st.button("💾 保存障碍物"):
            if save_obstacles_to_file(st.session_state.obstacles):
                st.sidebar.success("已保存")
    with col_save2:
        if st.button("📂 加载障碍物"):
            loaded = load_obstacles_from_file()
            if loaded:
                st.session_state.obstacles = loaded
                st.sidebar.success(f"加载 {len(loaded)} 个")
                st.rerun()
            else:
                st.sidebar.warning("无备份文件或文件损坏")
    if st.sidebar.button("🧹 清空所有障碍物"):
        st.session_state.obstacles = []
        if os.path.exists(OBSTACLE_FILE):
            os.remove(OBSTACLE_FILE)
        st.session_state.detour_route = None
        st.sidebar.success("已清空")
        st.rerun()
    if st.sidebar.button("🔄 重置应用"):
        st.session_state.obstacles = []
        if os.path.exists(OBSTACLE_FILE):
            os.remove(OBSTACLE_FILE)
        st.session_state.detour_route = None
        st.session_state.history = []
        st.session_state.sim = HeartbeatSimulator()
        st.rerun()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📍 坐标输入")
        lat_a = st.number_input("起点 A 纬度", value=32.2322, format="%.6f")
        lon_a = st.number_input("起点 A 经度", value=118.7490, format="%.6f")
        lat_b = st.number_input("终点 B 纬度", value=32.2343, format="%.6f")
        lon_b = st.number_input("终点 B 经度", value=118.7495, format="%.6f")
        flight_height = st.slider("设定飞行高度 (m)", 0, 100, 50)

        if coord_mode == "GCJ-02":
            display_lon_a, display_lat_a = gcj02_to_wgs84(lon_a, lat_a)
            display_lon_b, display_lat_b = gcj02_to_wgs84(lon_b, lat_b)
            st.success("已自动将 GCJ-02 坐标转换为 WGS-84")
        else:
            display_lon_a, display_lat_a = lon_a, lat_a
            display_lon_b, display_lat_b = lon_b, lat_b
            st.info("直接使用 WGS-84 坐标")

        if st.button("✈️ 生成可靠绕行航线"):
            with st.spinner("正在计算绕行路径（多障碍物自动避障）..."):
                A_wgs = (display_lon_a, display_lat_a)
                B_wgs = (display_lon_b, display_lat_b)
                detour = generate_detour_route(
                    A_wgs, B_wgs, 
                    st.session_state.obstacles, 
                    flight_height,
                    st.session_state.safety_distance
                )
                if len(detour) == 2:
                    st.success("✅ 无冲突，无需绕行")
                    st.session_state.detour_route = None
                else:
                    # 最终验证
                    if not path_intersects_any_obstacle(detour, st.session_state.obstacles, flight_height):
                        st.success(f"✅ 已生成可靠绕行航线，共 {len(detour)} 个航点")
                        st.session_state.detour_route = detour
                    else:
                        st.error("⚠️ 无法生成完全避障航线，请增加基础安全距离或减少障碍物")
                        st.session_state.detour_route = None
                st.rerun()

        if st.button("清除绕行航线"):
            st.session_state.detour_route = None
            st.rerun()

        if st.button("清除所有障碍物"):
            st.session_state.obstacles = []
            save_obstacles_to_file(st.session_state.obstacles)
            st.session_state.detour_route = None
            st.rerun()

    with col2:
        map_center = [display_lat_a, display_lon_a]
        m = folium.Map(
            location=map_center, zoom_start=17,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery',
        )

        folium.PolyLine(
            locations=[[display_lat_a, display_lon_a], [display_lat_b, display_lon_b]],
            color="yellow", weight=5, opacity=0.8, popup="原始航线"
        ).add_to(m)

        if st.session_state.get("detour_route"):
            detour_locs = [[lat, lng] for lng, lat in st.session_state.detour_route]
            folium.PolyLine(
                locations=detour_locs, color="blue", weight=4, opacity=0.9,
                popup="可靠绕行航线"
            ).add_to(m)
            start_pt = st.session_state.detour_route[0]
            end_pt = st.session_state.detour_route[-1]
            folium.Marker([start_pt[1], start_pt[0]], popup="绕行起点", icon=folium.Icon(color='blue', icon='play')).add_to(m)
            folium.Marker([end_pt[1], end_pt[0]], popup="绕行终点", icon=folium.Icon(color='blue', icon='stop')).add_to(m)

        folium.Marker([display_lat_a, display_lon_a], popup=f"起点 A (高度:{flight_height}m)", icon=folium.Icon(color='red', icon='play')).add_to(m)
        folium.Marker([display_lat_b, display_lon_b], popup="终点 B", icon=folium.Icon(color='green', icon='stop')).add_to(m)

        for idx, obs in enumerate(st.session_state.obstacles):
            poly_folium = [[lat, lng] for lng, lat in obs["vertices"]]
            folium.Polygon(
                locations=poly_folium, color="red", weight=3, fill=True, fill_color="red", fill_opacity=0.3,
                popup=f"障碍物 {idx+1}\n高度: {obs['height']} m"
            ).add_to(m)

        draw = Draw(
            draw_options={"polyline": False, "rectangle": True, "circle": False, "marker": False, "circlemarker": False, "polygon": True},
            edit_options={"edit": True, "remove": True}
        )
        draw.add_to(m)
        output = st_folium(m, width=800, height=500, returned_objects=["last_active_drawing"])

        if output and output.get("last_active_drawing"):
            drawing = output["last_active_drawing"]
            geom_type = drawing.get("geometry", {}).get("type")
            coords = drawing.get("geometry", {}).get("coordinates")
            if geom_type == "Polygon" and coords:
                ring = coords[0]
                poly_wgs84 = [(lng, lat) for lng, lat in ring]
                exists = any(obs["vertices"] == poly_wgs84 for obs in st.session_state.obstacles)
                if not exists:
                    new_obs = {"vertices": poly_wgs84, "height": st.session_state.default_obstacle_height}
                    st.session_state.obstacles.append(new_obs)
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.success(f"已添加障碍物（高度 {new_obs['height']} m）")
                    st.rerun()
            elif geom_type == "Rectangle" and coords:
                lng1, lat1 = coords[0]; lng2, lat2 = coords[1]
                rect = [(lng1, lat1), (lng2, lat1), (lng2, lat2), (lng1, lat2)]
                exists = any(obs["vertices"] == rect for obs in st.session_state.obstacles)
                if not exists:
                    new_obs = {"vertices": rect, "height": st.session_state.default_obstacle_height}
                    st.session_state.obstacles.append(new_obs)
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.success("已添加矩形障碍物")
                    st.rerun()

elif page == "飞行监控":
    st.header("✈️ 飞行监控 (心跳包实时状态)")
    placeholder = st.empty()
    if st.button("开始接收实时数据", key="btn_monitor_v20"):
        for _ in range(50):
            packet = st.session_state.sim.generate_packet()
            st.session_state.history.append(packet)
            plot_df = pd.DataFrame(st.session_state.history[-20:])
            with placeholder.container():
                m1, m2, m3 = st.columns(3)
                avg_rtt, loss_rate = st.session_state.sim.get_summary(st.session_state.history)
                m1.metric("实时 RTT", f"{packet['rtt']:.3f}s", delta=packet['status'], delta_color="inverse")
                m2.metric("平均 RTT", f"{avg_rtt:.3f}s")
                m3.metric("累计丢包率", f"{loss_rate:.1f}%")
                st.subheader("通讯延迟 (RTT) 变化曲线")
                st.line_chart(plot_df.set_index("time")["rtt"])
                if packet['is_timeout']:
                    st.error(f"警报：北京时间 {packet['time']} 发生通讯超时！")
            time.sleep(0.4)
    else:
        st.info("请点击按钮开始模拟监控。")
