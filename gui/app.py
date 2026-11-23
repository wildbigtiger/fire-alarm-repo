# -*- coding: utf-8 -*-
"""智能火灾报警系统 · 可视化原型（子任务 5）

启动：
    .\\.venv\\Scripts\\python.exe -m pip install -r requirements-gui.txt
    .\\.venv\\Scripts\\python.exe -m streamlit run gui\\app.py

界面分四块，对应题目要求：
  ① 实时检测  —— 画面+检测框+类别+置信度；火焰面积占比；三级报警
  ② 趋势与报警 —— 面积占比随时间曲线、报警等级历史、可检索
  ③ 历史回放  —— 调出历史事件的标注图/视频
  ④ 摄像头与定位 —— 水平角/俯仰角、多通道切换、GPS 电子地图打点

说明：
  * 摄像头角度与 GPS 按题目要求可"模拟实现"，这里用固定参数+可调滑块展示，
    逻辑是真实的（角度/经纬度参与展示与记录），只是数据来源是模拟值。
  * 分割掩码方面当前模型是检测模型，掩码显示区域预留，接 SAM 后填进去即可。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import common as C  # noqa: E402
from demo_infer import FireDetector, draw  # noqa: E402

OUT_DIR = ROOT / "runs" / "gui"
C.ensure_dir(OUT_DIR)

st.set_page_config(page_title="智能火灾报警系统", page_icon="🔥", layout="wide")

# ---------------------------------------------------------------- 模拟设备数据
CAMERAS = {
    "CAM-01 后厨": {"lat": 31.8612, "lon": 117.2830, "h": 35.0, "p": -8.0, "note": "室内 · 灶台"},
    "CAM-02 楼道": {"lat": 31.8615, "lon": 117.2836, "h": 120.0, "p": -12.0, "note": "室内 · 走廊"},
    "CAM-03 厂区": {"lat": 31.8602, "lon": 117.2821, "h": 200.0, "p": -20.0, "note": "室外 · 堆料场"},
}
LEVEL_COLOR = {"NORMAL": "#2e7d32", "ATTENTION": "#f9a825", "WARNING": "#ef6c00", "ALARM": "#c62828"}


@st.cache_resource(show_spinner=False)
def load_detector(weights: str, imgsz: int) -> FireDetector:
    return FireDetector(weights, imgsz=imgsz, device="0")


def init_state() -> None:
    if "events" not in st.session_state:
        st.session_state.events = []          # 历史记录
    if "last_img" not in st.session_state:
        st.session_state.last_img = None      # 最近一张标注图（numpy）
    if "series" not in st.session_state:
        st.session_state.series = []          # 逐帧曲线数据


init_state()


def resolve_weights() -> str:
    """按优先级找模型权重，兼容"开发目录"与"交付包"两种布局。"""
    cands = [
        ROOT / "runs" / "detect" / "baseline_v5" / "weights" / "best.pt",   # 训练产物目录
        ROOT.parent / "5_模型权重" / "baseline_v5" / "best.pt",              # 交付包里
        ROOT / "5_模型权重" / "baseline_v5" / "best.pt",
        ROOT / "weights" / "best.pt",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    for p in sorted(ROOT.parent.glob("**/best.pt")):
        return str(p)
    return str(cands[0])


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.title("🔥 火灾报警系统")
    weights = st.text_input("模型权重", value=resolve_weights(),
                            help="留空或路径不存在时请在下方提示处指定 best.pt 的位置")
    conf = st.slider("置信度阈值 conf", 0.05, 0.9, 0.25, 0.05)
    alarm_conf = st.slider(
        "报警置信度门槛", 0.20, 0.90, 0.50, 0.05,
        help="只有高于此值的框才计入“火灾面积占比”与报警等级。调低更敏感（易误报），调高更保守（易漏报）。",
    )
    imgsz = st.select_slider("输入分辨率", options=[320, 480, 640, 960], value=640)

    st.divider()
    cam = st.selectbox("摄像头通道", list(CAMERAS.keys()))
    meta = CAMERAS[cam]
    st.caption(f"{meta['note']} · 模拟设备")
    h_angle = st.slider("水平角 (°)", 0.0, 360.0, meta["h"], 1.0)
    p_angle = st.slider("俯仰角 (°)", -60.0, 30.0, meta["p"], 1.0)
    lat = st.number_input("纬度", value=meta["lat"], format="%.6f")
    lon = st.number_input("经度", value=meta["lon"], format="%.6f")

    st.divider()
    st.caption("报警分级（可在 demo_infer.py 里改）")
    st.caption("≥5% → 关注　≥15% 且持续 2s → 报警　≥30% → 立即报警")

# ---------------------------------------------------------------- 标题区
st.markdown("## 智能火灾报警系统 · 实时监测平台")
if not Path(weights).exists():
    st.error(
        f"找不到权重文件：{weights}\n\n"
        "**解决办法**：把左侧「模型权重」改成你的 `best.pt` 实际路径。\n\n"
        "若这是刚解压的交付包，权重通常在 `5_模型权重/baseline_v5/best.pt`，"
        "请填该文件的完整路径；也可以先运行 `scripts/train_yolo.py` 自行训练。"
    )
    st.stop()

det = load_detector(weights, imgsz)
det.alarm_conf = alarm_conf  # 侧边栏滑块实时生效

tab1, tab2, tab3, tab4 = st.tabs(["① 实时检测", "② 趋势与报警", "③ 历史回放", "④ 摄像头与定位"])

# ---------------------------------------------------------------- ① 实时检测
with tab1:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        src = st.radio("输入源", ["上传图片", "上传视频", "示例图片"], horizontal=True)
        up = None
        if src == "上传图片":
            up = st.file_uploader("选一张图", type=["jpg", "jpeg", "png", "bmp", "webp"])
        elif src == "上传视频":
            up = st.file_uploader("选一段视频", type=["mp4", "avi", "mov", "mkv"])
        else:
            demo_imgs = C.iter_images(C.RAW_IMAGES / "scene04_后厨明火")[:6]
            if demo_imgs:
                pick = st.selectbox("示例", [p.name for p in demo_imgs])
                up = next(p for p in demo_imgs if p.name == pick)

    run = st.button("▶ 开始检测", type="primary", use_container_width=True)

    with col_r:
        st.metric("当前通道", cam)
        st.metric("水平角 / 俯仰角", f"{h_angle:.0f}° / {p_angle:.0f}°")
        st.metric("GPS", f"{lat:.4f}, {lon:.4f}")

    if run and up is not None:
        import cv2
        import numpy as np

        is_video = isinstance(up, Path) is False and getattr(up, "name", "").lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )

        # ---- 图片 ----
        if not is_video:
            if isinstance(up, Path):
                frame = cv2.imread(str(up))
            else:
                arr = np.frombuffer(up.read(), np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("图片读不出来，换个格式试试")
            else:
                t0 = time.time()
                res = det.infer(frame, conf=conf)
                dt = (time.time() - t0) * 1000
                annotated = draw(frame.copy(), res)
                st.session_state.last_img = annotated
                st.image(annotated, channels="BGR", use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("火焰目标", res["counts"]["fire"])
                c2.metric("烟雾目标", res["counts"]["smoke"])
                c3.metric("火焰面积占比", f"{res['fire_area_ratio']*100:.1f}%")
                c4.metric("单帧耗时", f"{dt:.0f} ms")

                lv = res["alarm"]
                st.markdown(
                    f"<div style='background:{LEVEL_COLOR[lv['text']]};color:white;padding:12px;"
                    f"border-radius:8px;font-size:20px;font-weight:bold;text-align:center'>"
                    f"报警等级：{lv['text']}</div>",
                    unsafe_allow_html=True,
                )

                ev = {
                    "时间": datetime.now().strftime("%H:%M:%S"),
                    "通道": cam,
                    "来源": getattr(up, "name", "示例"),
                    "火": res["counts"]["fire"],
                    "烟": res["counts"]["smoke"],
                    "面积占比": round(res["fire_area_ratio"] * 100, 2),
                    "等级": lv["text"],
                    "水平角": h_angle,
                    "俯仰角": p_angle,
                    "纬度": lat,
                    "经度": lon,
                }
                st.session_state.events.append(ev)
                st.session_state.series.append(
                    {"t": len(st.session_state.series), "面积占比": res["fire_area_ratio"] * 100,
                     "等级": lv["level"]}
                )
                out = OUT_DIR / f"{datetime.now():%H%M%S}_{ev['来源']}.jpg"
                cv2.imwrite(str(out), annotated)
                st.caption(f"已保存：{out}")

        # ---- 视频 ----
        else:
            tmp = OUT_DIR / f"_{int(time.time())}_{up.name}"
            tmp.write_bytes(up.read())
            cap = cv2.VideoCapture(str(tmp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = OUT_DIR / f"{datetime.now():%H%M%S}_pred.mp4"
            vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            stride = max(1, int(fps // 5))  # 每秒处理 5 帧，保证流畅
            bar = st.progress(0.0, text="处理中…")
            placeholder = st.empty()
            i = done = 0
            alarm_frames = 0
            max_ratio = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                if i % stride == 0:
                    res = det.infer(frame, conf=conf)
                    vw.write(draw(frame, res))
                    done += 1
                    max_ratio = max(max_ratio, res["fire_area_ratio"])
                    if res["alarm"]["level"] >= 2:
                        alarm_frames += 1
                    st.session_state.series.append(
                        {"t": i / fps, "面积占比": res["fire_area_ratio"] * 100,
                         "等级": res["alarm"]["level"]}
                    )
                    if done % 5 == 0:
                        placeholder.image(draw(frame.copy(), res), channels="BGR",
                                          use_container_width=True)
                        bar.progress(min(1.0, i / total if total else 0), text=f"{i}/{total} 帧")
                else:
                    vw.write(frame)
            cap.release()
            vw.release()
            bar.progress(1.0, text="完成")
            st.video(str(out_path))
            st.success(f"处理 {done} 帧（抽帧处理），最大火焰占比 {max_ratio*100:.1f}%，"
                       f"预警以上 {alarm_frames} 帧 → {out_path}")
            st.session_state.events.append({
                "时间": datetime.now().strftime("%H:%M:%S"), "通道": cam, "来源": up.name,
                "火": "-", "烟": "-", "面积占比": round(max_ratio * 100, 2),
                "等级": "ALARM" if max_ratio >= 0.15 else ("WARNING" if max_ratio >= 0.05 else "NORMAL"),
                "水平角": h_angle, "俯仰角": p_angle, "纬度": lat, "经度": lon,
            })

# ---------------------------------------------------------------- ② 趋势与报警
with tab2:
    import pandas as pd

    st.subheader("火灾面积占比趋势")
    if st.session_state.series:
        df = pd.DataFrame(st.session_state.series)
        st.line_chart(df.set_index("t")["面积占比"], height=280, color="#c62828")
        st.caption("横轴：帧序号或时间（秒）；纵轴：火焰面积占比 %")
        st.subheader("报警等级变化")
        st.area_chart(df.set_index("t")["等级"], height=180)
    else:
        st.info("还没有数据。去「① 实时检测」跑一张图或一段视频。")

    st.subheader("报警历史记录")
    if st.session_state.events:
        st.dataframe(pd.DataFrame(st.session_state.events), use_container_width=True)
        if st.button("导出 CSV"):
            p = OUT_DIR / f"history_{datetime.now():%Y%m%d_%H%M%S}.csv"
            pd.DataFrame(st.session_state.events).to_csv(p, index=False, encoding="utf-8-sig")
            st.success(f"已导出 {p}")
    else:
        st.info("暂无记录。")

# ---------------------------------------------------------------- ③ 历史回放
with tab3:
    import pandas as pd

    st.subheader("历史报警片段检索与回放")
    if not st.session_state.events:
        st.info("先去「① 实时检测」产生几条记录。")
    else:
        df = pd.DataFrame(st.session_state.events)
        levels = st.multiselect("按等级筛选", sorted(df["等级"].unique()),
                                default=[l for l in df["等级"].unique() if l != "NORMAL"])
        view = df[df["等级"].isin(levels)] if levels else df
        st.dataframe(view, use_container_width=True)

        files = sorted(OUT_DIR.glob("*.jpg")) + sorted(OUT_DIR.glob("*.mp4"))
        if files:
            pick = st.selectbox("选择要回放的片段", [f.name for f in reversed(files)])
            p = OUT_DIR / pick
            if p.suffix == ".mp4":
                st.video(str(p))
            else:
                st.image(str(p), use_container_width=True)
            st.caption(f"{p} · {p.stat().st_size/1024:.0f} KB")
        else:
            st.info("还没有保存的片段。")

# ---------------------------------------------------------------- ④ 摄像头与定位
with tab4:
    import pandas as pd

    st.subheader("摄像头姿态")
    c1, c2, c3 = st.columns(3)
    c1.metric("水平角（偏航）", f"{h_angle:.1f}°")
    c2.metric("俯仰角", f"{p_angle:.1f}°")
    c3.metric("通道", cam)

    st.markdown(
        f"""
        <div style="display:flex;gap:20px;align-items:center;margin:8px 0 18px">
          <div style="width:180px;height:180px;border:2px solid #555;border-radius:50%;
                      position:relative;background:#111">
            <div style="position:absolute;left:50%;top:50%;width:2px;height:80px;background:#ff5252;
                        transform-origin:bottom;transform:translate(-50%,-100%) rotate({h_angle}deg)"></div>
            <div style="position:absolute;width:100%;text-align:center;bottom:6px;color:#bbb;
                        font-size:12px">水平 {h_angle:.0f}°</div>
          </div>
          <div style="width:180px;height:180px;border:2px solid #555;border-radius:50%;
                      position:relative;background:#111">
            <div style="position:absolute;left:50%;top:50%;width:2px;height:80px;background:#4fc3f7;
                        transform-origin:bottom;transform:translate(-50%,-100%) rotate({-p_angle}deg)"></div>
            <div style="position:absolute;width:100%;text-align:center;bottom:6px;color:#bbb;
                        font-size:12px">俯仰 {p_angle:.0f}°</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("多通道切换")
    rows = [{"通道": k, "位置": v["note"], "纬度": v["lat"], "经度": v["lon"],
             "水平角": v["h"], "俯仰角": v["p"]} for k, v in CAMERAS.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("GPS 定位与报警点")
    pts = [{"lat": v["lat"], "lon": v["lon"], "名称": k} for k, v in CAMERAS.items()]
    alarm_pts = [{"lat": e["纬度"], "lon": e["经度"]}
                 for e in st.session_state.events if e["等级"] in ("WARNING", "ALARM")]
    st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", size=120, color="#1e88e5")
    if alarm_pts:
        st.caption(f"报警点位 {len(alarm_pts)} 个")
        st.map(pd.DataFrame(alarm_pts), latitude="lat", longitude="lon", size=260, color="#c62828")
    else:
        st.caption("暂无报警点位（跑出 WARNING/ALARM 后会在这里打红点）")

    st.info("摄像头角度与 GPS 按题目说明采用模拟数据；展示与记录逻辑与真实传感器一致。")
