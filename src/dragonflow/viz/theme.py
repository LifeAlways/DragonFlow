"""DragonFlow 可视化主题 — 金融专业暗色风格。

配色灵感：同花顺 / 东方财富 / Bloomberg Terminal
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from pyecharts import options as opts

# =====================================================================
# 色板常量
# =====================================================================

COLORS = {
    # 背景
    "bg":           "#1a1a2e",
    "panel":        "#16213e",
    "grid":         "#2a2a4a",
    # 文字
    "text":         "#e0e0e0",
    "text_sub":     "#8892b0",
    # 涨跌
    "up":           "#ff4444",
    "down":         "#00d26a",
    # 强调
    "gold":         "#ffd700",
    "blue":         "#4fc3f7",
    "purple":       "#b388ff",
    "orange":       "#ffab40",
    "pink":         "#ff6e9c",
    "cyan":         "#80deea",
    # 中性
    "neutral":      "#546e7a",
    "white":        "#ffffff",
    "flat":         "#555555",
}

# 聚类色板（最多8个簇）
CLUSTER_PALETTE = [
    "#ff4444",  # 红
    "#4fc3f7",  # 蓝
    "#ffd700",  # 金
    "#00d26a",  # 绿
    "#b388ff",  # 紫
    "#ffab40",  # 橙
    "#ff6e9c",  # 粉
    "#80deea",  # 青
]

# 用于散点图等的渐变色
SCATTER_CMAP = "cool"  # 蓝紫渐变，暗色背景好看


# =====================================================================
# Matplotlib 暗色主题
# =====================================================================

def apply_dark_theme() -> None:
    """设置 Matplotlib 全局 rcParams 为金融暗色主题。"""
    bg = COLORS["bg"]
    text = COLORS["text"]
    grid = COLORS["grid"]

    plt.rcParams.update({
        # 字体
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "font.weight": "normal",
        # SVG：保持文本为 <text>，PDF 字体内嵌为 type 42（皆可编辑）
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        # 背景
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        # 文字颜色
        "text.color": text,
        "axes.labelcolor": text,
        "xtick.color": text,
        "ytick.color": text,
        # 网格
        "axes.grid": True,
        "grid.color": grid,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # 坐标轴
        "axes.edgecolor": grid,
        "axes.linewidth": 0.8,
        # 图例
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": grid,
        "legend.fontsize": 9,
        "legend.labelcolor": text,
        # 刻度
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        # 线条
        "lines.linewidth": 1.8,
        "lines.antialiased": True,
    })


# =====================================================================
# Pyecharts 暗色配置
# =====================================================================

PYECHARTS_BG = COLORS["bg"]
PYECHARTS_TEXT = COLORS["text"]
PYECHARTS_TEXT_SUB = COLORS["text_sub"]


def dark_init_opts(width: str = "1000px", height: str = "500px") -> opts.InitOpts:
    """返回暗色背景的 Pyecharts InitOpts。"""
    return opts.InitOpts(
        width=width,
        height=height,
        bg_color=PYECHARTS_BG,
        # theme 不用内置 dark，自定义更灵活
    )


def dark_title(title: str, subtitle: str = "") -> opts.TitleOpts:
    """暗色主题下的标题配置。"""
    return opts.TitleOpts(
        title=title,
        subtitle=subtitle,
        title_textstyle_opts=opts.TextStyleOpts(
            color=COLORS["gold"],
            font_size=16,
            font_weight="bold",
        ),
        subtitle_textstyle_opts=opts.TextStyleOpts(
            color=COLORS["text_sub"],
            font_size=11,
        ),
    )


def dark_legend(**kwargs) -> opts.LegendOpts:
    """暗色主题下的图例配置。"""
    defaults = {
        "textstyle_opts": opts.TextStyleOpts(color=COLORS["text"], font_size=10),
    }
    defaults.update(kwargs)
    return opts.LegendOpts(**defaults)


def dark_axis(name: str = "", is_scale: bool = False) -> opts.AxisOpts:
    """暗色主题下的坐标轴配置。"""
    return opts.AxisOpts(
        name=name,
        name_textstyle_opts=opts.TextStyleOpts(color=COLORS["text_sub"]),
        axislabel_opts=opts.LabelOpts(color=COLORS["text_sub"]),
        axisline_opts=opts.AxisLineOpts(
            linestyle_opts=opts.LineStyleOpts(color=COLORS["grid"])
        ),
        splitline_opts=opts.SplitLineOpts(
            is_show=True,
            linestyle_opts=opts.LineStyleOpts(color=COLORS["grid"], opacity=0.4),
        ),
        is_scale=is_scale,
    )


def dark_tooltip() -> opts.TooltipOpts:
    """暗色主题下的提示框配置。"""
    return opts.TooltipOpts(
        trigger="axis",
        background_color=COLORS["panel"],
        border_color=COLORS["grid"],
        textstyle_opts=opts.TextStyleOpts(color=COLORS["text"], font_size=12),
    )


def dark_datazoom() -> list:
    return [opts.DataZoomOpts(
        range_start=0, range_end=100,
        background_color=COLORS["panel"],
        fillercolor="rgba(79,195,247,0.15)",
        border_color=COLORS["grid"],
    )]
