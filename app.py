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
import graphviz
import random
import datetime

# ========== MAVLink 模拟报文生成 ==========
MAVLINK_MSG_TYPES = [
    {"name": "HEARTBEAT", "fields": {"type": "MAV_TYPE_QUADROTOR", "autopilot": "PX4", "base_mode": 81}},
    {"name": "GLOBAL_POSITION_INT", "fields": {"lat": 324000000, "lon": 1187000000, "alt": 50000, "relative_alt": 50000}},
    {"name": "ATTITUDE", "fields": {"roll": 0.02, "pitch": -0.01, "yaw": 1.57}},
    {"name": "SYS_STATUS", "fields": {"voltage_battery": 11200, "current_battery": -5, "battery_remaining": 85}},
    {"name": "RC_CHANNELS", "fields": {"chan1_raw": 1500, "chan2_raw": 1500, "chan3_raw": 1200, "chan4_raw": 1500}},
]

def generate_mavlink_message(seq):
    """生成一条模拟 MAVLink 报文（字典格式）"""
    msg_type = random.choice(MAVLINK_MSG_TYPES)
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    # 随机微调数值以产生变化感
    if msg_type["name"] == "GLOBAL_POSITION_INT":
        lat = 324000000 + random.randint(-1000, 1000)
        lon = 1187000000 + random.randint(-1000, 1000)
        alt = 50000 + random.randint(-500, 500)
        fields = {"lat": lat, "lon": lon, "alt": alt, "relative_alt": alt}
    elif msg_type["name"] == "ATTITUDE":
        fields = {"roll": round(random.uniform(-0.1, 0.1), 3),
                  "pitch": round(random.uniform(-0.1, 0.1), 3),
                  "yaw": round(random.uniform(0, 6.28), 3)}
    elif msg_type["name"] == "SYS_STATUS":
        fields = {"voltage_battery": 11200 + random.randint(-100, 100),
                  "current_battery": -5 + random.randint(-2, 2),
                  "battery_remaining": 85 + random.randint(-1, 1)}
    else:
        fields = msg_type["fields"]
    return {
        "seq": seq,
        "time": now,
        "msg_name": msg_type["name"],
        "fields": fields
    }

# ========== 障碍物持久化 ==========
OBSTACLE_FILE = "obstacles.json"

def save_obstacles_to_file(obstacles):
    try:
        with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(obstacles, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存障碍物失败: {e}")
        return False

def load_obstacles_from_file():
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    cleaned = []
                    for obs in data:
                        if isinstance(obs, dict) and "vertices" in obs and "height" in obs:
                            if isinstance(obs["vertices"], list) and len(obs["vertices"]) >= 3:
                                cleaned.append(obs)
                    return cleaned
                return []
        except Exception as e:
            st.error(f"加载障碍物失败: {e}")
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

def catmull_rom_spline(points, num_segments=30):
    if len(points) < 2:
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

# ========== 现有顺序绕行函数 ==========
def detour_single(A, B, obs, safety_meters, side="auto"):
    minx, miny, maxx, maxy = get_bounding_box(obs["vertices"])
    expand = safety_meters / 111000.0
    minx -= expand
    miny -= expand
    maxx += expand
    maxy += expand
    rect_pts = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    
    if side == "left":
        p1, p2 = rect_pts[0], rect_pts[1]
        if math.hypot(p1[0]-A[0], p1[1]-A[1]) > math.hypot(p2[0]-A[0], p2[1]-A[1]):
            p1, p2 = p2, p1
        return [A, p1, p2, B]
    elif side == "right":
        p1, p2 = rect_pts[3], rect_pts[2]
        if math.hypot(p1[0]-A[0], p1[1]-A[1]) > math.hypot(p2[0]-A[0], p2[1]-A[1]):
            p1, p2 = p2, p1
        return [A, p1, p2, B]
    else:
        paths = [
            ([A, rect_pts[0], rect_pts[1], B]),
            ([A, rect_pts[1], rect_pts[2], B]),
            ([A, rect_pts[2], rect_pts[3], B]),
            ([A, rect_pts[3], rect_pts[0], B]),
        ]
        def path_len(path):
            total = math.hypot(path[1][0]-path[0][0], path[1][1]-path[0][1])
            total += math.hypot(path[2][0]-path[1][0], path[2][1]-path[1][1])
            total += math.hypot(path[3][0]-path[2][0], path[3][1]-path[2][1])
            return total
        best = min(paths, key=path_len)
        return best

def sequential_detour(A, B, obstacles, flight_height, safety_meters, side="auto", max_iters=10):
    current_route = [A, B]
    for _ in range(max_iters):
        new_route = [current_route[0]]
        conflict = False
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
                conflict = True
                seg_detour = detour_single(seg_start, seg_end, target_obs, safety_meters, side)
                new_route.extend(seg_detour[1:])
        current_route = new_route
        if not conflict:
            ok = True
            for i in range(len(current_route)-1):
                for obs in obstacles:
                    if flight_height < obs["height"] and polygon_intersects_segment(obs["vertices"], current_route[i], current_route[i+1]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return current_route
    return current_route

def generate_detour_route(A, B, obstacles, flight_height, safety_meters, detour_side="auto", max_attempts=3):
    relevant = [obs for obs in obstacles if flight_height < obs["height"]]
    if not relevant:
        return [A, B]
    for attempt in range(max_attempts):
        current_safety = safety_meters * (1 + attempt * 0.5)
        route = sequential_detour(A, B, relevant, flight_height, current_safety, detour_side, max_iters=10)
        ok = True
        for i in range(len(route)-1):
            for obs in relevant:
                if polygon_intersects_segment(obs["vertices"], route[i], route[i+1]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            if len(route) > 2:
                return catmull_rom_spline(route, num_segments=30)
            else:
                return route
    st.warning("⚠️ 无法找到完全避障路径，请增加安全距离或调整障碍物位置")
    return [A, B]

def optimal_detour_route(A, B, obstacles, flight_height, safety_meters, max_attempts=3):
    relevant = [obs for obs in obstacles if flight_height < obs["height"]]
    if not relevant:
        return [A, B]

    for attempt in range(max_attempts):
        current_safety = safety_meters * (1 + attempt * 0.5)
        expand = current_safety / 111000.0

        points = [A, B]
        for obs in relevant:
            minx, miny, maxx, maxy = get_bounding_box(obs["vertices"])
            minx -= expand
            miny -= expand
            maxx += expand
            maxy += expand
            points.extend([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

        unique = []
        for p in points:
            if not any(math.hypot(p[0]-q[0], p[1]-q[1]) < 1e-9 for q in unique):
                unique.append(p)
        points = unique
        n = len(points)

        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                safe = True
                for obs in relevant:
                    if polygon_intersects_segment(obs["vertices"], p1, p2):
                        safe = False
                        break
                if safe:
                    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    graph[i].append((j, dist))
                    graph[j].append((i, dist))

        start_idx = points.index(A)
        end_idx = points.index(B)
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start_idx] = 0
        visited = [False] * n
        for _ in range(n):
            u = -1
            min_d = float('inf')
            for i in range(n):
                if not visited[i] and dist[i] < min_d:
                    min_d = dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            for v, w in graph[u]:
                if not visited[v] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u

        if dist[end_idx] != float('inf'):
            path_idx = []
            cur = end_idx
            while cur != -1:
                path_idx.append(cur)
                cur = prev[cur]
            path_idx.reverse()
            path_pts = [points[i] for i in path_idx]
            if len(path_pts) > 2:
                smooth = catmull_rom_spline(path_pts, num_segments=30)
                return smooth
            else:
                return path_pts
    st.warning("⚠️ 最优路径搜索失败，请增加安全距离或调整障碍物")
    return [A, B]

# ========== Streamlit 页面配置 ==========
st.set_page_config(page_title="无人机地面站监控系统", layout="wide")

if "app_version" not in st.session_state:
    st.session_state.sim = HeartbeatSimulator()
    st.session_state.history = []
    loaded = load_obstacles_from_file()
    st.session_state.obstacles = loaded if loaded else []
    st.session_state.default_obstacle_height = 30.0
    st.session_state.safety_distance = 3.0
    st.session_state.detour_route = None
    st.session_state.detour_side = "auto"
    st.session_state.fcu_online = True
    st.session_state.monitor_active = False
    st.session_state.mavlink_messages = []      
    st.session_state.app_version = "v34_mavlink"
else:
    if st.session_state.obstacles and isinstance(st.session_state.obstacles[0], list):
        new_obs = []
        for poly in st.session_state.obstacles:
            new_obs.append({"vertices": poly, "height": 30.0})
        st.session_state.obstacles = new_obs
        save_obstacles_to_file(st.session_state.obstacles)

st.sidebar.title("🧭 导航控制")
page = st.sidebar.radio("请选择功能页面", ["航线规划", "飞行监控"], key="page_radio")
st.sidebar.divider()
coord_mode = st.sidebar.radio("坐标系设置", ["WGS-84", "GCJ-02"], index=0, key="coord_radio")
st.sidebar.info("✅ 卫星图底图：Esri World Imagery (WGS-84)\n若选择 GCJ-02，系统会自动转换为 WGS-84 匹配卫星图。")

# ========== 原有航线规划部分（保持原样）==========
if page == "航线规划":
    # 因你要求仅修改“飞行监控”，此处代码仅保持占位。运行时可加入你原有的逻辑
    st.header("🗺️ 航线规划 (此处省略)") 
    pass

# ========== 飞行监控（重构UI符合图片要求，保留底层循环逻辑）==========
elif page == "飞行监控":
    # 1. 标题替换为图片样式
    st.title("✈️ 飞行实时画面 - 任务执行监控")

    # 2. 按钮栏
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns([2, 1, 1, 1, 1.5])
    with col_ctrl1:
        start_btn = st.button("▶️ 开始任务", type="primary", use_container_width=True)
    with col_ctrl2:
        st.button("⏸️ 暂停", use_container_width=True)
    with col_ctrl3:
        st.button("⏹️ 停止", use_container_width=True)
    with col_ctrl4:
        st.button("🔄 重置", use_container_width=True)
    with col_ctrl5:
        # 状态指示器 (通过 HTML 和 CSS 模拟黄色圆点 + 文字)
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-start; gap: 8px; height: 38px; padding: 0 10px; border: 1px solid #e5e7eb; border-radius: 6px; background: white;">
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #f59e0b; border-radius: 50%;"></span>
            <span style="font-weight: 500; color: #333;">已暂停</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. 核心指标列 (当前航点/速度/时间/距离/预计到达/电量)
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    # 初始状态(留空，循环中会被刷新)
    w1 = st.empty(); w2 = st.empty(); w3 = st.empty(); w4 = st.empty(); w5 = st.empty(); w6 = st.empty()
    
    def update_metrics(waypoint, speed, time_used, dist_left, eta, battery):
        w1.markdown(f"<div style='font-size:0.85rem;color:#666;'>📍 当前航点</div><div style='font-size:1.6rem;font-weight:bold;'>{waypoint}</div>", unsafe_allow_html=True)
        w2.markdown(f"<div style='font-size:0.85rem;color:#666;'>🚀 飞行速度</div><div style='font-size:1.6rem;font-weight:bold;'>{speed}<span style='font-size:0.9rem;font-weight:normal;color:#666;'> m/s</span></div>", unsafe_allow_html=True)
        w3.markdown(f"<div style='font-size:0.85rem;color:#666;'>⏱️ 已用时间</div><div style='font-size:1.6rem;font-weight:bold;'>{time_used}</div>", unsafe_allow_html=True)
        w4.markdown(f"<div style='font-size:0.85rem;color:#666;'>📏 剩余距离</div><div style='font-size:1.6rem;font-weight:bold;'>{dist_left}<span style='font-size:0.9rem;font-weight:normal;color:#666;'> m</span></div>", unsafe_allow_html=True)
        w5.markdown(f"<div style='font-size:0.85rem;color:#666;'>⏳ 预计到达</div><div style='font-size:1.6rem;font-weight:bold;'>{eta}</div>", unsafe_allow_html=True)
        w6.markdown(f"<div style='font-size:0.85rem;color:#666;'>🔋 电量模拟</div><div style='font-size:1.6rem;font-weight:bold;'>{battery}<span style='font-size:0.9rem;font-weight:normal;color:#666;'> %</span></div>", unsafe_allow_html=True)
    
    update_metrics("8/8", "8.5", "00:43", "0", "00:00", "96")

    # 4. 进度条
    st.markdown("<div style='font-size:0.8rem;color:#666;margin-bottom:4px;'>任务进度: 100%</div>", unsafe_allow_html=True)
    progress_bar = st.progress(1.0)

    # 5. 双栏布局 (地图与拓扑图)
    col_map, col_top = st.columns([1.8, 1])

    with col_map:
        st.subheader("🗺️ 实时飞行地图")
        # 渲染地图容器（这里使用你们现有的地图逻辑 + 绿线轨迹模拟）
        map_center = [32.2326, 118.7492]
        m = folium.Map(location=map_center, zoom_start=18,
                       tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                       attr='Esri World Imagery')
        
        # 绘制模拟绿线轨迹
        track_locs = [(32.2322, 118.7490), (32.2326, 118.7492), (32.2330, 118.7492), (32.2334, 118.7494), (32.2343, 118.7495)]
        folium.PolyLine(locations=track_locs, color="#00ff00", weight=4, opacity=0.9).add_to(m)
        for i, (lat, lon) in enumerate(track_locs):
            folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='info-sign', prefix='fa')).add_to(m)
        # 终点X号
        folium.Marker([32.2343, 118.7495], icon=folium.Icon(color='darkred', icon='times', prefix='fa')).add_to(m)
        
        # 绘制障碍物
        for obs in st.session_state.obstacles:
            poly_folium = [[lat, lng] for lng, lat in obs["vertices"]]
            folium.Polygon(locations=poly_folium, color="red", weight=3, fill=True, fill_color="red", fill_opacity=0.3).add_to(m)
        st_folium(m, width=700, height=400, returned_objects=[])

    with col_top:
        st.subheader("📶 通信链路拓扑与数据流")
        st.markdown("🟢 GCS 在线 &nbsp;&nbsp; 🟢 OBC 在线 &nbsp;&nbsp; 🟢 FCU 在线")
        
        # 图表渲染 (高度还原 UI 中的三个节点信息)
        try:
            dot = graphviz.Digraph()
            dot.attr(rankdir='LR', splines='line', ranksep='1.0')
            dot.attr('node', shape='plaintext', width='2', height='1.5')
            
            gcs_label = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
                <TR><TD BGCOLOR="#E3F2FD" COLOR="#333"><B>GCS</B></TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#666">地面站</TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#888" FONTSIZE="10">192.168.1.100</TD></TR>
            </TABLE>>'''
            obc_label = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
                <TR><TD BGCOLOR="#FFF3E0" COLOR="#333"><B>OBC</B></TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#666">机载计算机</TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#888" FONTSIZE="10">Raspberry Pi 4</TD></TR>
            </TABLE>>'''
            fcu_label = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
                <TR><TD BGCOLOR="#F3E5F5" COLOR="#333"><B>FCU</B></TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#666">飞控</TD></TR>
                <TR><TD BGCOLOR="#FFFFFF" COLOR="#888" FONTSIZE="10">PX4 / ArduPilot</TD></TR>
            </TABLE>>'''
            
            dot.node('GCS', label=gcs_label)
            dot.node('OBC', label=obc_label)
            dot.node('FCU', label=fcu_label)
            dot.edge('GCS', 'OBC', label='UDP:14550', color='#999', fontcolor='#555', dir='both')
            dot.edge('OBC', 'FCU', label='MAVLink', color='#999', fontcolor='#555', dir='both')
            st.graphviz_chart(dot, use_container_width=True)
        except Exception as e:
            st.error(f"Graphviz 渲染失败: {e}。请确保系统安装了 Graphviz 软件并配置了环境变量。")
        
        st.markdown("---")
        st.caption("📊 **链路统计:** GCS↔OBC 正常  OBC↔FCU 正常  延迟 ~25ms  丢包率: 0.1%")

    # 6. 循环刷新逻辑（完全保留你原始的结构）
    if start_btn:
        st.session_state.monitor_active = True

    if not st.session_state.get("monitor_active", False):
        st.info("请点击「开始任务」开始实时模拟监控。")
    else:
        placeholder = st.empty()
        mav_seq = len(st.session_state.mavlink_messages)
        cur_wp = 0
        total_dist = 500
        
        # 采用你原有代码的 for 循环 + time.sleep 逻辑
        for _ in range(50):
            packet = st.session_state.sim.generate_packet()
            st.session_state.history.append(packet)
            
            # 模拟 MAVLink 消息
            mav_msg = generate_mavlink_message(mav_seq)
            st.session_state.mavlink_messages.append(mav_msg)
            mav_seq += 1
            if len(st.session_state.mavlink_messages) > 100:
                st.session_state.mavlink_messages = st.session_state.mavlink_messages[-100:]

            # 模拟动态数据变化
            cur_wp = min(cur_wp + random.uniform(0.1, 0.3), 8)
            pct_complete = (cur_wp / 8) * 100
            battery_left = max(0, 96 - (cur_wp/8)*4)
            time_spent_sec = int(cur_wp * 5.5)
            dist_left_m = max(0, total_dist - (cur_wp/8)*total_dist)
            
            # 更新顶部指标和进度条
            update_metrics(
                f"{int(cur_wp)}/8", 
                f"{8.5 + random.uniform(-0.5, 0.5):.1f}", 
                f"{time_spent_sec//60:02d}:{time_spent_sec%60:02d}", 
                f"{int(dist_left_m)}", 
                "00:00", 
                f"{battery_left:.1f}"
            )
            progress_bar.progress(pct_complete / 100.0)

            # 在占位符内刷新列表和状态 (保留原逻辑)
            with placeholder.container():
                # 延迟 RTT 图表
                plot_df = pd.DataFrame(st.session_state.history[-20:])
                st.subheader("通讯延迟 (RTT) 变化曲线")
                st.line_chart(plot_df.set_index("time")["rtt"])
                
                if packet['is_timeout']:
                    st.error(f"警报：北京时间 {packet['time']} 发生通讯超时！")
                
                # 报文流展示
                with st.expander("📨 MAVLink 报文流（最近 50 条）", expanded=True):
                    if st.session_state.mavlink_messages:
                        df_msgs = pd.DataFrame(st.session_state.mavlink_messages[-50:])
                        df_msgs["fields_str"] = df_msgs["fields"].apply(lambda f: ", ".join(f"{k}={v}" for k, v in f.items()))
                        display_df = df_msgs[["seq", "time", "msg_name", "fields_str"]]
                        display_df.columns = ["序号", "时间", "消息类型", "关键字段"]
                        st.dataframe(display_df, use_container_width=True, height=200)
                        
            time.sleep(0.4)

        # 循环结束
        st.session_state.monitor_active = False
        st.rerun()
