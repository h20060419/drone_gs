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

# ========== 常量与辅助函数 ==========
OBSTACLE_FILE = "obstacles.json"
EPS = 1e-9

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
        ret += (20.0 * math.sin(lng * PI) + 20.0 * math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
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

# ========== 几何辅助函数（与原代码相同，为节省篇幅省略中间部分，请确保完整粘贴）==========
# ... （此处应粘贴您原有的所有几何函数，包括 segments_intersect, point_on_segment, 
#     get_bounding_box, polygon_intersects_segment, path_intersects_any_obstacle, 
#     line_intersection_point, catmull_rom_spline, detour_around_obstacle_robust, 
#     generate_detour_route 等，因篇幅限制不重复写，实际使用时请保留原代码完整）
# ========== 重要：请确保以下函数完整存在 ==========
# 为了代码可运行，这里简略，但您必须保留原文件中的全部几何函数

# ========== Streamlit 页面配置 ==========
st.set_page_config(page_title="无人机地面站监控系统", layout="wide")

if "app_version" not in st.session_state or st.session_state.app_version != "v22_fixed":
    st.session_state.sim = HeartbeatSimulator()
    st.session_state.history = []
    loaded = load_obstacles_from_file()
    st.session_state.obstacles = loaded if loaded else []
    st.session_state.default_obstacle_height = 30.0
    st.session_state.safety_distance = 3.0
    st.session_state.detour_route = None
    st.session_state.pending_obstacle = None
    st.session_state.monitoring_active = False
    st.session_state.app_version = "v22_fixed"
else:
    if st.session_state.obstacles:
        st.session_state.obstacles = validate_obstacles(st.session_state.obstacles)

st.sidebar.title("🧭 导航控制")
page = st.sidebar.radio("请选择功能页面", ["航线规划", "飞行监控"])
st.sidebar.divider()
coord_mode = st.sidebar.radio("坐标系设置（输入坐标的原始类型）", ["WGS-84", "GCJ-02"], index=0)
st.sidebar.info("✅ 卫星图底图：Esri World Imagery (WGS-84)\n若选择 GCJ-02，系统会自动转换为 WGS-84 匹配卫星图。")

# ========== 页面：航线规划 ==========
if page == "航线规划":
    st.header("🗺️ 航线规划 + 障碍物圈选 (多障碍物可靠绕行)")

    # 醒目提示：待确认障碍物机制
    st.sidebar.warning("⚠️ 地图上绘制多边形/矩形后，需在下方点击【✅ 确认添加】才会生效！")

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

    # 强制忽略高度复选框（调试用）
    ignore_flight_height = st.sidebar.checkbox("✈️ 强制避让所有障碍物（忽略飞行高度）", value=False,
                                               help="开启后，无论飞行高度如何，都会尝试绕行所有障碍物")

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
        st.session_state.pending_obstacle = None
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

    # 主要规划区域
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
            with st.spinner("正在计算绕行路径..."):
                A_wgs = (display_lon_a, display_lat_a)
                B_wgs = (display_lon_b, display_lat_b)

                # 决定需要避让的障碍物列表
                if ignore_flight_height:
                    obstacles_to_avoid = st.session_state.obstacles.copy()
                    st.info("已开启强制避让模式，将尝试绕行所有障碍物（忽略飞行高度）")
                else:
                    obstacles_to_avoid = [obs for obs in st.session_state.obstacles if flight_height < obs["height"]]
                    if not obstacles_to_avoid:
                        st.success(f"当前飞行高度 {flight_height}m 大于等于所有障碍物高度，无需绕行。")
                        st.session_state.detour_route = None
                        st.rerun()
                    else:
                        st.info(f"需要避让 {len(obstacles_to_avoid)} 个障碍物（高度低于飞行高度）")

                # 检测原始直线是否与需要避让的障碍物相交
                raw_conflict = path_intersects_any_obstacle([A_wgs, B_wgs], obstacles_to_avoid, flight_height=None)  # 不传flight_height，检测所有
                if not raw_conflict:
                    st.success("✅ 原始航线不穿过任何需避让的障碍物，无需绕行")
                    st.session_state.detour_route = None
                    st.rerun()

                # 需要绕行
                detour = generate_detour_route(
                    A_wgs, B_wgs, 
                    obstacles_to_avoid,   # 只传给需要避让的障碍物
                    flight_height if not ignore_flight_height else 0,   # 若强制模式，传0使所有障碍物都视为需避让
                    st.session_state.safety_distance
                )

                # 验证绕行结果
                if len(detour) == 2:
                    # 返回直线，检查是否真的无冲突（理论上应该冲突，因为上面已判断raw_conflict=True）
                    st.error("❌ 绕行算法无法找到可行路径！请尝试增加安全距离、减少障碍物或调整起点/终点。")
                    st.session_state.detour_route = None
                else:
                    # 检查绕行路线是否仍与障碍物相交
                    if not path_intersects_any_obstacle(detour, obstacles_to_avoid, flight_height=None):
                        st.success(f"✅ 已生成可靠绕行航线，共 {len(detour)} 个航点")
                        st.session_state.detour_route = detour
                    else:
                        st.error("⚠️ 绕行后路径仍与障碍物相交，请增加安全距离或简化障碍物形状。")
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

        # 待确认障碍物处理
        if st.session_state.pending_obstacle:
            st.subheader("⏳ 待确认的障碍物")
            st.write(f"顶点数: {len(st.session_state.pending_obstacle['vertices'])}")
            col_confirm, col_cancel = st.columns(2)
            if col_confirm.button("✅ 确认添加"):
                st.session_state.obstacles.append(st.session_state.pending_obstacle)
                save_obstacles_to_file(st.session_state.obstacles)
                st.session_state.pending_obstacle = None
                st.rerun()
            if col_cancel.button("❌ 取消"):
                st.session_state.pending_obstacle = None
                st.rerun()

    with col2:
        # 地图绘制部分与原代码完全相同，此处省略，请保留您原有的地图代码
        # ... （为了完整性，您需将原来的地图绘制代码粘贴在此）
        # 注意：地图部分保持不变即可
        pass

# ========== 页面：飞行监控（与原代码相同，可保留）==========
elif page == "飞行监控":
    st.header("✈️ 飞行监控 (心跳包实时状态)")
    placeholder = st.empty()
    start_btn = st.button("开始接收实时数据", key="start_monitor")
    stop_btn = st.button("停止监控", key="stop_monitor")

    if start_btn:
        st.session_state.monitoring_active = True
    if stop_btn:
        st.session_state.monitoring_active = False

    if st.session_state.monitoring_active:
        for _ in range(200):
            if not st.session_state.monitoring_active:
                break
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
                if packet.get('is_timeout', False):
                    st.error(f"警报：北京时间 {packet['time']} 发生通讯超时！")
            time.sleep(0.4)
            st.rerun()
        st.session_state.monitoring_active = False
        st.info("监控已自动停止（达到循环上限），可再次点击开始。")
    else:
        st.info("请点击「开始接收实时数据」启动模拟监控。")
