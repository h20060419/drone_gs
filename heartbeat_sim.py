import time
import random
from datetime import datetime, timedelta, timezone

class HeartbeatSimulator:
    def __init__(self, timeout_threshold=3.0):
        """
        初始化心跳模拟器
        """
        self.timeout_threshold = timeout_threshold
        self.seq = 0
        
    def generate_packet(self):
        """
        生成单次心跳数据包
        """
        # 1. 序号自增
        self.seq += 1
        
        # 2. 获取北京时间 (UTC+8) [针对作业要求的横坐标显示]
        tz_utc_8 = timezone(timedelta(hours=8))
        time_str = datetime.now(tz_utc_8).strftime("%H:%M:%S")
        
        # 3. 核心修复：提前定义变量默认值，防止任何分支下的 NameError
        rtt = 0.0
        is_timeout = False
        send_time = time.time()
        
        # 4. 模拟网络逻辑
        rand_val = random.random()
        
        if rand_val < 0.8:
            # 80% 正常情况
            rtt = random.uniform(0.01, 0.05)
            is_timeout = False
        elif rand_val < 0.95:
            # 15% 网络抖动
            rtt = random.uniform(0.1, 0.8)
            is_timeout = False
        else:
            # 5% 丢包/超时
            rtt = self.timeout_threshold + random.uniform(0.1, 0.5)
            is_timeout = True
            
        # 5. 返回确保包含所有 key 的字典
        return {
            "seq": self.seq,
            "time": time_str,     # 对应 app.py 中的 set_index("time")
            "send_time": send_time,
            "rtt": rtt,           # 确保此处 rtt 已经被定义
            "is_timeout": is_timeout,
            "status": "正常" if not is_timeout else "超时警告"
        }

    def get_summary(self, history):
        """
        统计历史数据
        """
        if not history:
            return 0.0, 0.0
            
        total = len(history)
        timeouts = sum(1 for p in history if p.get('is_timeout', False))
        loss_rate = (timeouts / total) * 100
        
        # 提取有效的 RTT
        valid_rtts = [p['rtt'] for p in history if not p.get('is_timeout', False)]
        avg_rtt = sum(valid_rtts) / len(valid_rtts) if valid_rtts else 0.0
        
        return avg_rtt, loss_rate