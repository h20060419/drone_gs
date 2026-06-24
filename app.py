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

# ========== Dijkstra 最优路径 ==========
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

# ---------- 初始化 session_state ----------
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
    # 监控控制状态
    st.session_state.monitor_running = False
    st.session_state.monitor_paused = False
    # MAVLink 与实时地图
    st.session_state.mavlink_messages = []
    st.session_state.drone_position = (32.2322, 118.7490)  # 初始坐标
    st.session_state.drone_track = []
    st.session_state.app_version = "v36_realtimemap_controls"
else:
    # 旧版本障碍物格式迁移（保留）
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

# ========== 航线规划页面（原样保留）==========
if page == "航线规划":
    # ... 航线规划页面代码与 v34 完全一致，此处省略重复内容 ...
    pass   # 在完整代码中请保留原来的航线规划全部代码

# ========== 飞行监控页面（新增控制与实时地图）==========
elif page == "飞行监控":
    st.header("✈️ 飞行监控 (实时地图 + 通信状态)")

    # ---------- 控制栏与状态指示灯 ----------
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5, col_ctrl6 = st.columns([1, 1, 1, 1, 1, 2])
    with col_ctrl1:
        if st.button("▶️ 开始任务", key="btn_start", use_container_width=True):
            st.session_state.monitor_running = True
            st.session_state.monitor_paused = False
            st.rerun()
    with col_ctrl2:
        if st.session_state.monitor_running and not st.session_state.monitor_paused:
            if st.button("⏸️ 暂停", key="btn_pause", use_container_width=True):
                st.session_state.monitor_paused = True
                st.rerun()
        elif st.session_state.monitor_paused:
            if st.button("▶️ 继续", key="btn_resume", use_container_width=True):
                st.session_state.monitor_paused = False
                st.rerun()
        else:
            st.button("⏸️ 暂停", key="btn_pause_disabled", disabled=True, use_container_width=True)
    with col_ctrl3:
        if st.button("⏹️ 停止", key="btn_stop", use_container_width=True):
            st.session_state.monitor_running = False
            st.session_state.monitor_paused = False
            st.rerun()
    with col_ctrl4:
        if st.button("🔄 重置", key="btn_reset", use_container_width=True):
            st.session_state.monitor_running = False
            st.session_state.monitor_paused = False
            st.session_state.history = []
            st.session_state.mavlink_messages = []
            st.session_state.drone_position = (32.2322, 118.7490)
            st.session_state.drone_track = []
            st.rerun()
    with col_ctrl5:
        # 模拟 FCU 故障按钮
        if st.button("⚠️ FCU 故障", key="toggle_fcu", use_container_width=True):
            st.session_state.fcu_online = not st.session_state.fcu_online
            st.rerun()
    with col_ctrl6:
        # 状态指示灯
        if st.session_state.monitor_running:
            if st.session_state.monitor_paused:
                status_color = "#FFA500"   # 橙色
                status_text = "已暂停"
            else:
                status_color = "#00FF00"   # 绿色
                status_text = "飞行中"
        else:
            status_color = "#808080"       # 灰色
            status_text = "已停止"
        st.markdown(
            f"<div style='display:flex; align-items:center;'>"
            f"<span style='background-color:{status_color}; width:16px; height:16px; border-radius:50%; display:inline-block; margin-right:8px;'></span>"
            f"<span style='font-weight:bold;'>{status_text}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ---------- 实时地图（最上方）----------
    st.subheader("📍 无人机实时位置")
    # 每次重新渲染地图（即使暂停也保持显示）
    drone_map = folium.Map(
        location=[st.session_state.drone_position[0], st.session_state.drone_position[1]],
        zoom_start=17,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
    )
    # 绘制轨迹
    if len(st.session_state.drone_track) >= 2:
        track_locs = [[lat, lon] for lat, lon in st.session_state.drone_track]
        folium.PolyLine(
            locations=track_locs,
            color="yellow", weight=3, opacity=0.8,
            popup="飞行轨迹"
        ).add_to(drone_map)
    # 当前位置标记
    folium.Marker(
        [st.session_state.drone_position[0], st.session_state.drone_position[1]],
        popup="无人机",
        icon=folium.Icon(color='red', icon='plane', prefix='fa')
    ).add_to(drone_map)
    # 动态 key 强制刷新
    st_folium(drone_map, width=800, height=400, key="drone_realtime_map")

    # ---------- 数据更新（仅当运行且未暂停时）----------
    if st.session_state.monitor_running and not st.session_state.monitor_paused:
        # 生成新数据
        packet = st.session_state.sim.generate_packet()
        st.session_state.history.append(packet)
        # 生成 MAVLink 报文
        mav_seq = len(st.session_state.mavlink_messages)
        mav_msg = generate_mavlink_message(mav_seq)
        st.session_state.mavlink_messages.append(mav_msg)
        if len(st.session_state.mavlink_messages) > 100:
            st.session_state.mavlink_messages = st.session_state.mavlink_messages[-100:]

        # 更新无人机位置
        if mav_msg["msg_name"] == "GLOBAL_POSITION_INT":
            lat = mav_msg["fields"]["lat"] / 1e7
            lon = mav_msg["fields"]["lon"] / 1e7
            st.session_state.drone_position = (lat, lon)
            st.session_state.drone_track.append((lat, lon))
            if len(st.session_state.drone_track) > 20:
                st.session_state.drone_track = st.session_state.drone_track[-20:]

    # ---------- 显示其他监控组件（无论是否暂停都显示最新数据）----------
    if st.session_state.history:   # 有数据才显示
        plot_df = pd.DataFrame(st.session_state.history[-20:])
        m1, m2, m3 = st.columns(3)
        avg_rtt, loss_rate = st.session_state.sim.get_summary(st.session_state.history)
        latest = st.session_state.history[-1]
        m1.metric("实时 RTT", f"{latest['rtt']:.3f}s",
                  delta=latest['status'], delta_color="inverse")
        m2.metric("平均 RTT", f"{avg_rtt:.3f}s")
        m3.metric("累计丢包率", f"{loss_rate:.1f}%")

        st.subheader("通讯延迟 (RTT) 变化曲线")
        st.line_chart(plot_df.set_index("time")["rtt"])

        if latest['is_timeout']:
            st.error(f"警报：北京时间 {latest['time']} 发生通讯超时！")
    else:
        st.info("暂无监控数据，请点击「开始任务」")

    # 拓扑图（始终显示）
    st.subheader("📡 GCS-OBC-FCU 通信拓扑")
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', size='6,2')
    gcs_color = 'lightblue'
    obc_color = 'lightblue'
    fcu_color = 'lightgreen' if st.session_state.fcu_online else 'lightcoral'
    dot.node('GCS', 'GCS\n(地面站)', shape='box', style='filled', fillcolor=gcs_color)
    dot.node('OBC', 'OBC\n(机载计算机)', shape='box', style='filled', fillcolor=obc_color)
    dot.node('FCU', 'FCU\n(飞控)', shape='box', style='filled', fillcolor=fcu_color)

    if st.session_state.history and not st.session_state.history[-1]['is_timeout']:
        rtt_label = f"{st.session_state.history[-1]['rtt']:.3f} s"
        dot.edge('GCS', 'OBC', label=rtt_label, color='green', fontcolor='green')
    else:
        dot.edge('GCS', 'OBC', label='超时', color='red', style='dashed', fontcolor='red')
    if st.session_state.fcu_online:
        dot.edge('OBC', 'FCU', label='0.005 s', color='green', fontcolor='green')
    else:
        dot.edge('OBC', 'FCU', label='中断', color='red', style='dashed', fontcolor='red')
    st.graphviz_chart(dot, use_container_width=True)

    # MAVLink 报文流
    with st.expander("📨 MAVLink 报文流（最近 50 条）", expanded=True):
        if st.session_state.mavlink_messages:
            df_msgs = pd.DataFrame(st.session_state.mavlink_messages[-50:])
            df_msgs["fields_str"] = df_msgs["fields"].apply(
                lambda f: ", ".join(f"{k}={v}" for k, v in f.items())
            )
            display_df = df_msgs[["seq", "time", "msg_name", "fields_str"]]
            display_df.columns = ["序号", "时间", "消息类型", "关键字段"]
            st.dataframe(display_df, use_container_width=True, height=300)
        else:
            st.info("暂无报文数据")

    # ---------- 循环驱动 ----------
    if st.session_state.monitor_running:
        time.sleep(0.4)
        st.rerun()
