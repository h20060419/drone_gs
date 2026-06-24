import streamlit as st
import pandas as pd
import time
import math
import json
import os
import random
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ==================== 简易心跳模拟器（内置） ====================
class HeartbeatSimulator:
    def __init__(self):
        self.base_rtt = 0.02
        self.packet_count = 0

    def generate_packet(self):
        self.packet_count += 1
        rtt = self.base_rtt + random.uniform(0, 0.05)
        if random.random() < 0.1:
            rtt += 0.3
        timeout = random.random() < 0.05
        status = "timeout" if timeout else "normal"
        return {
            "time": time.strftime("%H:%M:%S"),
            "rtt": rtt,
            "status": status,
            "is_timeout": timeout
        }

    def get_summary(self, history):
        if not history:
            return 0, 0
        avg_rtt = sum(p['rtt'] for p in history) / len(history)
        loss = sum(1 for p in history if p['is_timeout']) / len(history) * 100
        return avg_rtt, loss

# ==================== 持久化与状态初始化 ====================
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

# ==================== 初始化 session_state ====================
if "app_version" not in st.session_state:
    st.session_state.sim = HeartbeatSimulator()
    st.session_state.history = []
    st.session_state.obstacles = load_obstacles_from_file()
    st.session_state.default_obstacle_height = 30.0
    st.session_state.safety_distance = 3.0
    st.session_state.detour_route = None
    st.session_state.detour_side = "auto"
    st.session_state.mav_messages = []
    st.session_state.topology_status = {"gcs_obc": "up", "obc_fcu": "up"}
    st.session_state.app_version = "v34_plotly_full"
else:
    # 兼容旧格式障碍物
    if st.session_state.obstacles and isinstance(st.session_state.obstacles[0], list):
        new_obs = []
        for poly in st.session_state.obstacles:
            new_obs.append({"vertices": poly, "height": 30.0})
        st.session_state.obstacles = new_obs
        save_obstacles_to_file(st.session_state.obstacles)

# ==================== GCJ-02 转 WGS-84 ====================
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

# ==================== 几何辅助函数 ====================
def segments_intersect(x1,y1,x2,y2,x3,y3,x4,y4):
    def cross(ax,ay,bx,by): return ax*by-ay*bx
    def on_seg(px,py,qx,qy,rx,ry): return min(px,qx)<=rx<=max(px,qx) and min(py,qy)<=ry<=max(py,qy)
    o1=cross(x2-x1,y2-y1,x3-x1,y3-y1)
    o2=cross(x2-x1,y2-y1,x4-x1,y4-y1)
    o3=cross(x4-x3,y4-y3,x1-x3,y1-y3)
    o4=cross(x4-x3,y4-y3,x2-x3,y2-y3)
    if o1==0 and on_seg(x1,y1,x2,y2,x3,y3): return True
    if o2==0 and on_seg(x1,y1,x2,y2,x4,y4): return True
    if o3==0 and on_seg(x3,y3,x4,y4,x1,y1): return True
    if o4==0 and on_seg(x3,y3,x4,y4,x2,y2): return True
    return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)

def polygon_intersects_segment(poly_vertices, seg_start, seg_end):
    try:
        n=len(poly_vertices)
        if n<3: return False
        for i in range(n):
            x1,y1=poly_vertices[i]
            x2,y2=poly_vertices[(i+1)%n]
            if segments_intersect(seg_start[0],seg_start[1],seg_end[0],seg_end[1],x1,y1,x2,y2):
                return True
        mid_x=(seg_start[0]+seg_end[0])/2
        mid_y=(seg_start[1]+seg_end[1])/2
        inside=False
        for i in range(n):
            x1,y1=poly_vertices[i]
            x2,y2=poly_vertices[(i+1)%n]
            if ((y1>mid_y)!=(y2>mid_y)) and (mid_x<(x2-x1)*(mid_y-y1)/(y2-y1)+x1):
                inside=not inside
        return inside
    except:
        return False

def get_bounding_box(poly_vertices):
    xs=[v[0] for v in poly_vertices]
    ys=[v[1] for v in poly_vertices]
    return min(xs),min(ys),max(xs),max(ys)

def catmull_rom_spline(points, num_segments=30):
    if len(points)<2: return points
    if len(points)==2: return [points[0]+(points[1]-points[0])*t for t in [i/num_segments for i in range(num_segments+1)]]
    result=[]
    for i in range(len(points)-1):
        p0=points[max(i-1,0)]
        p1=points[i]
        p2=points[i+1]
        p3=points[min(i+2,len(points)-1)]
        for t in [j/num_segments for j in range(num_segments)]:
            t2=t*t; t3=t2*t
            x=0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y=0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((x,y))
    result.append(points[-1])
    return result

# ==================== 安全距离换算（修正经度缩放） ====================
def meters_to_lng_deg(meters, lat):
    return meters / (111320.0 * math.cos(math.radians(lat)))

def meters_to_lat_deg(meters):
    return meters / 110540.0

# ==================== 绕行算法 ====================
def detour_single(A, B, obs, safety_meters, side="auto"):
    minx,miny,maxx,maxy=get_bounding_box(obs["vertices"])
    avg_lat=(miny+maxy)/2
    expand_lng=meters_to_lng_deg(safety_meters, avg_lat)
    expand_lat=meters_to_lat_deg(safety_meters)
    minx-=expand_lng; miny-=expand_lat; maxx+=expand_lng; maxy+=expand_lat
    rect_pts=[(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)]
    if side=="left":
        p1,p2=rect_pts[0],rect_pts[1]
        if math.hypot(p1[0]-A[0],p1[1]-A[1])>math.hypot(p2[0]-A[0],p2[1]-A[1]): p1,p2=p2,p1
        return [A,p1,p2,B]
    elif side=="right":
        p1,p2=rect_pts[3],rect_pts[2]
        if math.hypot(p1[0]-A[0],p1[1]-A[1])>math.hypot(p2[0]-A[0],p2[1]-A[1]): p1,p2=p2,p1
        return [A,p1,p2,B]
    else:
        paths=[([A,rect_pts[0],rect_pts[1],B]),([A,rect_pts[1],rect_pts[2],B]),
               ([A,rect_pts[2],rect_pts[3],B]),([A,rect_pts[3],rect_pts[0],B])]
        def path_len(p): return math.hypot(p[1][0]-p[0][0],p[1][1]-p[0][1])+math.hypot(p[2][0]-p[1][0],p[2][1]-p[1][1])+math.hypot(p[3][0]-p[2][0],p[3][1]-p[2][1])
        best=min(paths,key=path_len)
        return best

def sequential_detour(A,B,obstacles,flight_height,safety_meters,side="auto",max_iters=10):
    route=[A,B]
    for _ in range(max_iters):
        new=[route[0]]
        conflict=False
        for i in range(len(route)-1):
            s=route[i]; e=route[i+1]
            obs_found=None
            for obs in obstacles:
                if flight_height<obs["height"] and polygon_intersects_segment(obs["vertices"],s,e):
                    obs_found=obs; break
            if obs_found is None:
                new.append(e)
            else:
                conflict=True
                d=detour_single(s,e,obs_found,safety_meters,side)
                new.extend(d[1:])
        route=new
        if not conflict:
            ok=True
            for i in range(len(route)-1):
                for obs in obstacles:
                    if flight_height<obs["height"] and polygon_intersects_segment(obs["vertices"],route[i],route[i+1]):
                        ok=False; break
                if not ok: break
            if ok: return route
    return route

def generate_detour_route(A,B,obstacles,flight_height,safety_meters,detour_side="auto",max_attempts=3):
    relevant=[obs for obs in obstacles if flight_height<obs["height"]]
    if not relevant: return [A,B]
    for attempt in range(max_attempts):
        cur_safety=safety_meters*(1+attempt*0.5)
        route=sequential_detour(A,B,relevant,flight_height,cur_safety,detour_side,10)
        ok=True
        for i in range(len(route)-1):
            for obs in relevant:
                if polygon_intersects_segment(obs["vertices"],route[i],route[i+1]):
                    ok=False; break
            if not ok: break
        if ok:
            if len(route)>2: return catmull_rom_spline(route,30)
            else: return route
    st.warning("⚠️ 无法找到完全避障路径，请增加安全距离或调整障碍物位置")
    return [A,B]

def optimal_detour_route(A,B,obstacles,flight_height,safety_meters,max_attempts=3):
    relevant=[obs for obs in obstacles if flight_height<obs["height"]]
    if not relevant: return [A,B]
    for attempt in range(max_attempts):
        cur_safety=safety_meters*(1+attempt*0.5)
        points=[A,B]
        for obs in relevant:
            minx,miny,maxx,maxy=get_bounding_box(obs["vertices"])
            avg_lat=(miny+maxy)/2
            expand_lng=meters_to_lng_deg(cur_safety,avg_lat)
            expand_lat=meters_to_lat_deg(cur_safety)
            minx-=expand_lng; miny-=expand_lat; maxx+=expand_lng; maxy+=expand_lat
            points.extend([(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)])
        unique=[]
        for p in points:
            if not any(math.hypot(p[0]-q[0],p[1]-q[1])<1e-9 for q in unique):
                unique.append(p)
        points=unique
        n=len(points)
        graph=[[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                p1=points[i]; p2=points[j]
                safe=True
                for obs in relevant:
                    if polygon_intersects_segment(obs["vertices"],p1,p2):
                        safe=False; break
                if safe:
                    d=math.hypot(p2[0]-p1[0],p2[1]-p1[1])
                    graph[i].append((j,d))
                    graph[j].append((i,d))
        start=points.index(A); end=points.index(B)
        dist=[float('inf')]*n; prev=[-1]*n; dist[start]=0
        visited=[False]*n
        for _ in range(n):
            u=-1; min_d=float('inf')
            for i in range(n):
                if not visited[i] and dist[i]<min_d: min_d=dist[i]; u=i
            if u==-1: break
            visited[u]=True
            for v,w in graph[u]:
                if not visited[v] and dist[u]+w<dist[v]:
                    dist[v]=dist[u]+w; prev[v]=u
        if dist[end]!=float('inf'):
            path=[]
            cur=end
            while cur!=-1: path.append(cur); cur=prev[cur]
            path.reverse()
            pts=[points[i] for i in path]
            if len(pts)>2: return catmull_rom_spline(pts,30)
            else: return pts
    st.warning("⚠️ 最优路径搜索失败，请增加安全距离或调整障碍物")
    return [A,B]

# ==================== 通信链路辅助函数（Plotly 拓扑图） ====================
def simulate_mavlink_message():
    msg_types = ["HEARTBEAT", "GLOBAL_POSITION_INT", "ATTITUDE", "SYS_STATUS", "BATTERY_STATUS"]
    msg_type = random.choice(msg_types)
    fields = {
        "type": msg_type,
        "time_boot_ms": random.randint(1000,5000),
        "lat": 32.23+random.uniform(-0.01,0.01),
        "lon": 118.75+random.uniform(-0.01,0.01),
        "alt": random.uniform(50,100),
        "roll": random.uniform(-30,30),
        "pitch": random.uniform(-20,20),
        "yaw": random.uniform(0,360),
        "voltage": random.uniform(11.1,12.6)
    }
    return {
        "time": time.strftime("%H:%M:%S"),
        "type": msg_type,
        "raw": "FE 09 00 00 00 01 01 00 00 00 00 00 00 00",
        "fields": fields
    }

def update_topology_status():
    if st.session_state.mav_messages:
        if random.random()<0.9:
            st.session_state.topology_status = {"gcs_obc":"up","obc_fcu":"up"}
        else:
            st.session_state.topology_status["obc_fcu"] = random.choice(["up","down"])
            st.session_state.topology_status["gcs_obc"] = "up" if random.random()<0.8 else "down"

def render_topology_plotly():
    """使用 Plotly 绘制 GCS-OBC-FCU 拓扑图"""
    status = st.session_state.topology_status
    nodes = {
        "GCS": (0, 1),
        "OBC": (1, 1),
        "FCU": (2, 1)
    }
    edges = [
        ("GCS", "OBC", status["gcs_obc"]),
        ("OBC", "FCU", status["obc_fcu"]),
    ]
    node_x = [nodes[n][0] for n in nodes]
    node_y = [nodes[n][1] for n in nodes]
    node_text = list(nodes.keys())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=40, color=['lightblue','lightyellow','lightgreen']),
        hoverinfo='text'
    )
    edge_traces = []
    for src, dst, state in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        color = 'green' if state == 'up' else 'red'
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=3, color=color),
            hoverinfo='none'
        ))
    layout = go.Layout(
        title="GCS-OBC-FCU 通信链路拓扑",
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 2.5]),
        yaxis=dict(visible=False, range=[0.5, 1.5]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    return fig

# ==================== Streamlit 页面配置 ====================
st.set_page_config(page_title="无人机地面站监控系统", layout="wide")
st.sidebar.title("🧭 导航控制")
page = st.sidebar.radio("请选择功能页面", ["航线规划", "飞行监控", "通信链路"])
st.sidebar.divider()
coord_mode = st.sidebar.radio("坐标系设置", ["WGS-84", "GCJ-02"], index=0)
st.sidebar.info("✅ 卫星图底图：Esri World Imagery (WGS-84)\n若选择 GCJ-02，系统会自动转换为 WGS-84 匹配卫星图。")

# ==================== 航线规划页面 ====================
if page == "航线规划":
    st.header("🗺️ 航线规划 + 多障碍物可靠绕行 (左侧/右侧/自动/最优)")

    st.sidebar.subheader("🚧 障碍物默认高度")
    default_h = st.sidebar.number_input(
        "新绘制障碍物的默认高度 (米)", 
        min_value=0.0, max_value=200.0, 
        value=st.session_state.default_obstacle_height, step=5.0,
        key="default_height"
    )
    st.session_state.default_obstacle_height = default_h
    st.sidebar.divider()

    st.sidebar.subheader("🛡️ 安全距离 (米)")
    safety = st.sidebar.number_input(
        "绕行安全距离", 
        min_value=0.0, max_value=200.0, 
        value=st.session_state.safety_distance, step=5.0,
        help="绕行路径与障碍物的最小距离（若找不到路径会自动增加）",
        key="safety_dist"
    )
    st.session_state.safety_distance = safety
    st.sidebar.divider()

    st.sidebar.subheader("↪️ 全局绕行侧偏好（仅对下方“自动绕行”有效）")
    side_option = st.sidebar.selectbox(
        "偏好绕行侧",
        options=["auto", "left", "right"],
        index=["auto", "left", "right"].index(st.session_state.detour_side),
        format_func=lambda x: {"auto": "自动选择最短路径", "left": "强制从左侧绕过", "right": "强制从右侧绕过"}[x],
        key="side_select"
    )
    st.session_state.detour_side = side_option
    st.sidebar.divider()

    st.sidebar.subheader("📋 已添加的障碍物")
    if not st.session_state.obstacles:
        st.sidebar.write("暂无障碍物")
    else:
        for idx, obs in enumerate(st.session_state.obstacles):
            with st.sidebar.expander(f"障碍物 {idx+1} (高度: {obs['height']} m)"):
                new_height = st.number_input(
                    f"高度 (m)", min_value=0.0, max_value=200.0, value=obs['height'],
                    key=f"obs_height_{idx}", step=5.0
                )
                if new_height != obs['height']:
                    obs['height'] = new_height
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.rerun()
                if st.button(f"🗑️ 删除障碍物 {idx+1}", key=f"del_obs_{idx}"):
                    st.session_state.obstacles.pop(idx)
                    save_obstacles_to_file(st.session_state.obstacles)
                    st.session_state.detour_route = None
                    st.rerun()
                st.caption(f"顶点数: {len(obs['vertices'])}")
    st.sidebar.metric("障碍物总数", len(st.session_state.obstacles))
    st.sidebar.divider()
    col_save1, col_save2 = st.sidebar.columns(2)
    with col_save1:
        if st.button("💾 保存障碍物", key="save_btn"):
            if save_obstacles_to_file(st.session_state.obstacles):
                st.sidebar.success("已保存")
    with col_save2:
        if st.button("📂 加载障碍物", key="load_btn"):
            loaded = load_obstacles_from_file()
            if loaded:
                st.session_state.obstacles = loaded
                st.sidebar.success(f"加载 {len(loaded)} 个")
                st.rerun()
            else:
                st.sidebar.warning("无备份文件或文件损坏")
    if st.sidebar.button("🧹 清空所有障碍物", key="clear_all"):
        st.session_state.obstacles = []
        if os.path.exists(OBSTACLE_FILE):
            os.remove(OBSTACLE_FILE)
        st.session_state.detour_route = None
        st.sidebar.success("已清空")
        st.rerun()
    if st.sidebar.button("🔄 重置应用", key="reset_all"):
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
        lat_a = st.number_input("起点 A 纬度", value=32.2322, format="%.6f", key="lat_a")
        lon_a = st.number_input("起点 A 经度", value=118.7490, format="%.6f", key="lon_a")
        lat_b = st.number_input("终点 B 纬度", value=32.2343, format="%.6f", key="lat_b")
        lon_b = st.number_input("终点 B 经度", value=118.7495, format="%.6f", key="lon_b")
        flight_height = st.slider("设定飞行高度 (m)", 0, 100, 50, key="flight_h")

        if coord_mode == "GCJ-02":
            display_lon_a, display_lat_a = gcj02_to_wgs84(lon_a, lat_a)
            display_lon_b, display_lat_b = gcj02_to_wgs84(lon_b, lat_b)
            st.success("已自动将 GCJ-02 坐标转换为 WGS-84")
        else:
            display_lon_a, display_lat_a = lon_a, lat_a
            display_lon_b, display_lat_b = lon_b, lat_b
            st.info("直接使用 WGS-84 坐标")

        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            if st.button("✈️ 自动绕行", key="btn_auto", use_container_width=True):
                with st.spinner("正在计算自动绕行路径..."):
                    A_wgs = (display_lon_a, display_lat_a)
                    B_wgs = (display_lon_b, display_lat_b)
                    route = generate_detour_route(
                        A_wgs, B_wgs,
                        st.session_state.obstacles,
                        flight_height,
                        st.session_state.safety_distance,
                        detour_side=st.session_state.detour_side
                    )
                    if len(route) == 2:
                        st.success("✅ 无冲突，无需绕行")
                        st.session_state.detour_route = None
                    else:
                        st.success(f"✅ 已生成自动绕行航线，共 {len(route)} 个航点")
                        st.session_state.detour_route = route
                    st.rerun()
        with col_btn2:
            if st.button("⬅️ 左侧绕行", key="btn_left", use_container_width=True):
                with st.spinner("正在计算左侧绕行路径..."):
                    A_wgs = (display_lon_a, display_lat_a)
                    B_wgs = (display_lon_b, display_lat_b)
                    route = generate_detour_route(
                        A_wgs, B_wgs,
                        st.session_state.obstacles,
                        flight_height,
                        st.session_state.safety_distance,
                        detour_side="left"
                    )
                    if len(route) == 2:
                        st.success("✅ 无冲突，无需绕行")
                        st.session_state.detour_route = None
                    else:
                        st.success(f"✅ 已生成左侧绕行航线，共 {len(route)} 个航点")
                        st.session_state.detour_route = route
                    st.rerun()
        with col_btn3:
            if st.button("➡️ 右侧绕行", key="btn_right", use_container_width=True):
                with st.spinner("正在计算右侧绕行路径..."):
                    A_wgs = (display_lon_a, display_lat_a)
                    B_wgs = (display_lon_b, display_lat_b)
                    route = generate_detour_route(
                        A_wgs, B_wgs,
                        st.session_state.obstacles,
                        flight_height,
                        st.session_state.safety_distance,
                        detour_side="right"
                    )
                    if len(route) == 2:
                        st.success("✅ 无冲突，无需绕行")
                        st.session_state.detour_route = None
                    else:
                        st.success(f"✅ 已生成右侧绕行航线，共 {len(route)} 个航点")
                        st.session_state.detour_route = route
                    st.rerun()
        with col_btn4:
            if st.button("🏆 最优路径", key="btn_optimal", use_container_width=True):
                with st.spinner("正在计算全局最优最短路径..."):
                    A_wgs = (display_lon_a, display_lat_a)
                    B_wgs = (display_lon_b, display_lat_b)
                    route = optimal_detour_route(
                        A_wgs, B_wgs,
                        st.session_state.obstacles,
                        flight_height,
                        st.session_state.safety_distance
                    )
                    if len(route) == 2:
                        st.success("✅ 无冲突，无需绕行")
                        st.session_state.detour_route = None
                    else:
                        st.success(f"✅ 已生成最优路径航线，共 {len(route)} 个航点")
                        st.session_state.detour_route = route
                    st.rerun()

        if st.button("清除绕行航线", key="clear_route"):
            st.session_state.detour_route = None
            st.rerun()

        if st.button("清除所有障碍物", key="clear_obs"):
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
                popup="绕行航线"
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

# ==================== 飞行监控页面 ====================
elif page == "飞行监控":
    st.header("✈️ 飞行监控 (心跳包实时状态)")
    placeholder = st.empty()
    col_start, col_stop = st.columns(2)
    with col_start:
        start_monitor = st.button("开始接收实时数据", key="monitor_start")
    with col_stop:
        stop_monitor = st.button("停止监控", key="monitor_stop")

    if 'monitor_active' not in st.session_state:
        st.session_state.monitor_active = False

    if start_monitor:
        st.session_state.monitor_active = True
    if stop_monitor:
        st.session_state.monitor_active = False

    if st.session_state.monitor_active:
        status_text = st.empty()
        while st.session_state.monitor_active:
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
            status_text.text(f"监控中... 最后更新: {packet['time']}")
        st.session_state.monitor_active = False
        st.rerun()
    else:
        st.info("请点击按钮开始模拟监控。")

# ==================== 通信链路页面 ====================
elif page == "通信链路":
    st.header("📡 通信链路监控 (GCS-OBC-FCU)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("拓扑图")
        topo_placeholder = st.empty()
        if st.button("刷新拓扑状态"):
            update_topology_status()
        topo_placeholder.plotly_chart(render_topology_plotly(), use_container_width=True)
        st.caption("绿色连线表示链路正常，红色表示中断")

    with col2:
        st.subheader("MAVLink 报文模拟")
        if st.button("生成一条模拟报文"):
            msg = simulate_mavlink_message()
            st.session_state.mav_messages.append(msg)
            update_topology_status()
        if st.button("清空报文"):
            st.session_state.mav_messages = []
        if st.session_state.mav_messages:
            df = pd.DataFrame(st.session_state.mav_messages)
            st.metric("总报文数", len(df))
            type_counts = df['type'].value_counts()
            st.bar_chart(type_counts)
            st.subheader("最新20条")
            st.dataframe(df[['time','type','raw']].tail(20), use_container_width=True)
            if st.checkbox("显示详细字段（最新）"):
                st.json(st.session_state.mav_messages[-1]['fields'])
        else:
            st.info("暂无报文，请点击生成模拟报文。")
