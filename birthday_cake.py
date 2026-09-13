# -*- coding: utf-8 -*-
"""
Happy Birthday —— 3D 粒子生日蛋糕
================================================================================
技术栈 : Python 3.10+ / pygame / PyOpenGL / numpy
绘制   : VBO + glVertexPointer / glColorPointer / glDrawArrays(GL_POINTS)
混合   : glBlendFunc(GL_SRC_ALPHA, GL_ONE) 加法混合
辉光   : FBO 离屏渲染 → 亮度阈值提取 → 小半径可分离高斯模糊(多次偏移累加)
         → 叠加回屏幕。强度低(0.4)，粒子本身保持锐利不被模糊。
说明   : 全程固定管线，不使用 GLSL 着色器，老显卡也能跑。

操作说明
    鼠标左键拖动   →  旋转视角
    鼠标滚轮       →  缩放视角
    ESC / 关闭窗口 →  退出
    （调试）python birthday_cake.py --shot   → 渲染 90 帧存 preview.png 后退出
================================================================================
"""

import math
import sys

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE,
    MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
)
from OpenGL.GL import *


# ==============================================================================
# ① 配置区 —— 调参数改这里
# ==============================================================================

# ------------------------------ 窗口 ------------------------------
WIN_W, WIN_H = 1000, 800
FPS = 60
TITLE = "Happy Birthday"
# 背景不是黑的。把 1.png 的四个角和左右两条边都采一遍，稳定落在
# (0.11, 0.12, 0.13) 上下 —— 一层抬起来的蓝灰，四个角几乎没有差别，
# 说明这不是"某处的光晕"，而是**整张图的底色**。
# 原先给 (0.02, 0.02, 0.06)：比参考图暗了七倍，蛋糕是浮在一片纯黑里的，
# 和参考图那种"夜里有一层薄雾"完全不是一回事。
# 抬底色还有个副作用是好的：蛋糕周围那些加法粒子（星云、纸屑）
# 本来就是"叠在底色上"的，底色一黑，它们只能靠自己的亮度硬撑，
# 看着是一颗颗孤立的光点；底色抬起来之后它们才连成一片雾。
# 0.10 这个值卡在 bloom 阈值（0.55 / 增益 1.2 = 0.458）之下很远，
# 底色不会自己发光。
BG_COLOR = (0.098, 0.104, 0.125, 1.0)

# ------------------------------ 粒子数量 ------------------------------
CAKE_COUNT = 82000         # 蛋糕主体（单层圆柱，深蓝 + 金大理石纹）。
                           # 数量不能省：圆柱正对相机的那一片投影密度天然最低
                           # （越靠两侧轮廓越是斜着看，投影越密），
                           # 粒子不够就在正中间漏出一片背景黑洞
PLATE_COUNT = 28000        # 底盘（浅色盘子）
RING_COUNT = 95000         # 光环。参考图里环是一条**连续**的亮线，点数不够就断成
                           # 虚线，一定要给足。62000 的时候放大看是串珠子：
                           # 环上粒子的线密度是固定的，一提按、带子一宽，
                           # 同样多的粒子摊到更宽的面积上，就稀到看得见颗粒了。
                           # 提到 95000 才连成一笔。这一项跟 RING_WIDTH 是一对 ——
                           # 以后再把带宽调粗，点数得跟着加
STAR_COUNT = 1500           # 背景星星。参考图的背景很空，铺满了反而像雪花屏
FLAME_COUNT = 5200         # 火焰
CANDLE_COUNT = 700         # 蜡烛
CONFETTI_COUNT = 3000       # 彩色纸屑（大小不一的彩点，左上方一簇）。
                           # 参考图里这一片是很"满"的：大大小小几十颗虚焦的圆点，
                           # 400 颗撒在同样大的范围里显得稀稀拉拉，像背景噪点而不像纸屑
SPARK_COUNT = 460          # 四角星形闪光
NEBULA_COUNT = 900         # 背景星云雾（大颗粒、低透明度，铺在蛋糕周围）

# ------------------------------ 粒子大小（像素） ------------------------------
# 每颗粒子在自己的区间里随机取大小，取整分档（分档才能合批，不然几千次 draw call）
CAKE_SIZE_RANGE = (5.0, 9.0)         # 蛋糕粒子（方点）。粒子越小轮廓的台阶越细，
                                     # 但太小就有像素盖不到、露出黑点；
                                     # 5~9 像素在"轮廓够平"和"铺得满"之间
PLATE_SIZE_RANGE = (4.0, 9.0)        # 底盘粒子要够大才铺得满，小了就成了一圈点
RING_SIZE_RANGE = (2.0, 4.0)         # 光环粒子（差别小才连得成一条线）
FLAME_SIZE_RANGE = (2.0, 5.0)        # 火焰粒子
CANDLE_POINT_SIZE = 4.0              # 蜡烛是方点。柱子只有 0.055 半径，点太小就铺不满
STAR_SIZE_RANGE = (1.0, 2.0)         # 星星 1~2 像素
CONFETTI_SIZE_RANGE = (4.0, 17.0)    # 纸屑大小差别要拉大，才有参考图里那种疏密错落。
                                     # 上限给到 17：参考图里的彩点是**虚焦**的，
                                     # 大的那几颗直径有小半个蛋糕那么宽，
                                     # 全挤在 3~11 就只是一层均匀的小圆点
SPARK_SIZE_RANGE = (9.0, 22.0)       # 四角星闪光（贴图是星形）
NEBULA_SIZE_RANGE = (20.0, 50.0)     # 背景星云雾颗粒

# 点精灵贴图：把方形的点变成"中间实、边缘渐隐"的柔和小光球。
# 贴图是程序算出来的，没有任何外部图片。
GLOW_TEX_SIZE = 32
GLOW_CORE = 0.38                     # 实心核心占半径的比例
GLOW_HALO = 1.40                     # 边缘柔光的衰减指数（越大边缘越锐）

# 四角星闪光贴图（参考图里散落的金色小星星）
STAR_TEX_SIZE = 32                   # 32 而不是 64：贴图越小，缩小采样时
                                     # 丢掉的细节越少（下面星芒的宽度说明）
STAR_TEX_CORE = 0.09                 # 中心实心核比例。核一大就只剩一个圆点，
                                     # 四角星要"核小芒长"才认得出是星星
STAR_TEX_ARM = 0.12                  # 星芒的角半宽（归一化，越小越细）。
                                     # 别低于 0.10：星芒是**细**特征，
                                     # 缩小时会被 mipmap 平均掉。
                                     # 原先 0.055 配 64 的贴图 —— 采样到的 mip 上
                                     # 星芒只剩不到一个像素，加上线性衰减，
                                     # 星芒直接没了，90 颗"四角星"全渲染成了小圆点。
STAR_TEX_FALLOFF = 1.5               # 星芒沿半径的衰减

# ------------------------------ "实体面"粒子 ------------------------------
# 蛋糕 / 盘子 / 光环 / 蜡烛这四组叫"实体面"，它们和发光的小点走**完全不同**的路：
#
# 1) 方点、不贴图。柔光球和圆盘都是"方贴图上切一个圆"，四角透明，
#    而点的深度值是**整颗点**一个值：角上的像素照样被写入深度、后面的粒子被挡掉，
#    于是角上什么也没画 —— 约 21% 的像素是洞，表面成了电视机雪花。
#    试过 alpha 测试（丢弃的片元不写深度）：洞是没了，但角上的像素会漏给
#    后面好几颗粒子叠上去，叠几颗全看运气，表面变成一块块白斑。
#    角要么挡住别人（成洞），要么不挡（成叠），两条路都难看。
#    干净的解法只有一个：让粒子**铺满自己的方块**。
#
# 2) 不透明混合（GL_ONE / GL_ZERO），并且把 alpha 预先乘进颜色里。
#    加法混合下，深度测试挡掉的是"比已画过的更远"的粒子；
#    也就是说，一颗粒子只要比**此前画过的**都近，它就会叠上去。
#    一个像素上大约 30 颗粒子互相覆盖，其中"一路刷新最近纪录"的期望有 ln(30)≈3.4 颗，
#    于是每个像素叠了三四层，整个蛋糕过曝成白色。
#    改成覆盖式写：只有严格更近的粒子才画得上去，最终每个像素留下的就是
#    唯一那颗最近的粒子，颜色就是大理石纹的颜色 —— 干净、稳定、和绘制顺序无关。
#
# 代价是实体不再有柔光。但参考图里的蛋糕体本来就是一块哑光的石纹面，
# 光晕交给 bloom 去做（火焰、光环、纸屑照旧是加法混合的发光小点）。

# ------------------------------ 蛋糕几何 ------------------------------
# 对照 1.png：单层矮圆柱 + 底下垫一块浅色圆盘（盘子）。
# 之前是上下两层，参考图是一层，这里改成一层。
CAKE_R, CAKE_H, CAKE_Y = 1.15, 1.68, -1.05     # 半径 / 高度 / 底面 y
CAKE_TOP_Y = CAKE_Y + CAKE_H                   # 顶面 y = 0.45
# 表面粒子占比。现在蛋糕体是不透明的，内部粒子永远看不见、只会添乱，
# 所以全部粒子都给表面（生成函数里已经没有"内部"那一档了）。
CAKE_SURFACE_RATIO = 1.0
CAKE_TOP_DENSITY = 0.40                        # 顶面粒子的密度折扣（见生成函数的说明）

# 底盘：比蛋糕略大的扁圆盘，边缘一圈微微翘起（盘子边）
PLATE_R, PLATE_Y = 1.52, -1.16
PLATE_RIM_H = 0.075                            # 盘子边缘翘起的高度

# ------------------------------ 蛋糕颜色 ------------------------------
# 这两个蓝是**照着 1.png 采样定的**，不是凭感觉挑的。
# 参考图柱身中位数 RGB = (0.153, 0.239, 0.384)（已归一化）：红绿都不低，
# 是个偏"灰"的藏青，不是纯蓝。原先给 (0.02, 0.05, 0.40)，
# 渲染出来中位数 (0.047, 0.082, 0.325) —— 红只有参考图的 1/3，
# 整根柱子看着比原图更暗、更"电"，像荧光蓝而不是印在纸上的深蓝。
# 反过来推（扣掉辉光抬起来的那约 0.07，再除竖直渐变和 alpha 的 0.63），
# 底面该在 (0.13, 0.27, 0.49) 附近。取七成，别一次给满 ——
# 参考图是小尺寸压缩图，本身带一点的糊，全按它调会过头。
COL_ROYAL_BLUE = (0.075, 0.145, 0.40)  # 蛋糕体主色：藏青（参考图的底色）
COL_MID_BLUE = (0.26, 0.44, 1.00)      # 亮蓝（云斑的亮面）。
                                       # **两者的蓝通道要拉开**（0.36 ↔ 0.95），
                                       # 花纹才起得来；红绿按比例一起抬，
                                       # 否则两种蓝只差在红绿上，看着是一片发乳的白蓝
# 石面上的云母闪点：暖白为主，掺金、淡蓝、粉。
# 不给纯白 —— 纯白在蓝底上会看成噪点，带一点暖色才像石头里的反光。
SPECK_COLORS = (
    (1.00, 0.93, 0.74),   # 暖白（主）
    (1.00, 0.93, 0.74),
    (1.00, 0.93, 0.74),
    (1.00, 0.82, 0.36),   # 金
    (0.80, 0.88, 1.00),   # 淡蓝
    (1.00, 0.76, 0.92),   # 粉
)
CAKE_SPECK_RATIO = 0.024             # 闪点占侧面粒子的比例。
                                     # 上一档是 0.015，柱身看着还是偏"哑"；
                                     # 但也不能给到 5% 以上 —— 那就成了电视雪花，
                                     # 宝蓝的底子会被亮点糊掉
COL_GOLD = (1.00, 0.80, 0.10)        # 金
COL_GOLD_LIGHT = (0.90, 0.66, 0.24)  # 浅金（漩涡纹用）。
                                     # 绿通道别给到 0.85：那是柠檬黄，
                                     # 参考图里的金脉是琥珀色，偏橙不偏黄。
                                     # 整体也不能给满：乘以竖直渐变之后，
                                     # 满亮的金会顶到 0.9 上下、远远超过 bloom 阈值，
                                     # 辉光再叠一层就成了荧光黄 —— 参考图里
                                     # 金脉是**闷**在蓝底子里的一条，不是一道光
# 顶面颜色要够"黄"。加法叠起来之后三通道一起涨，偏白的奶油金会直接顶成纯白，
# 只有把蓝通道压下去、保住色饱和，叠多少层都还是金的。
# 顶面同样是采样定的：参考图顶面中位数 (0.784, 0.671, 0.671) ——
# 红绿蓝**几乎相等**，是接近白的暖奶油色，蓝只比红低 15%。
# 原先给 (1.00, 0.92, 0.70)，蓝比红低 30%，出来是黄的；
# 叠上辉光红通道先顶到 1.0，屏幕上就是一块烧过头的金饼。
COL_CREAM = (1.00, 0.94, 0.90)       # 顶面：暖白（参考图里最亮的一片）
COL_CREAM_EDGE = (1.00, 0.78, 0.44)
COL_CREAM_GOLD = (1.00, 0.76, 0.30)  # 顶面漩涡用的浅金。
                                     # 不能直接用柱身那张 COL_GOLD_LIGHT(0.90,0.66,0.24)：
                                     # 那张是给蓝底子用的，压在奶白上就是一道道深琥珀，
                                     # 顶面会被切碎成土黄。参考图顶面上的金是**浅**的，
                                     # 亮到几乎和奶白底连成一片，只是色相偏暖  # 顶面外沿收的那道金。原来 0.66/0.20 是纯橙，
                                     # 一圈下来顶面就成了甜甜圈
COL_PLATE = (0.64, 0.68, 0.86)       # 底盘：淡薰衣草白。
                                     # 之前压到 0.62/0.68/0.92 是因为盘子看着像一块白甜甜圈，
                                     # 但那个"白"其实是盘沿和径向加码堆出来的，不是盘面本身亮。
                                     # 现在盘面拉平了，颜色可以正常给 —— 参考图里
                                     # 盘子就是整张图第二亮的东西，仅次于顶面
COL_PLATE_EDGE = (0.74, 0.75, 0.92)  # 盘子边缘亮一点，做出盘沿的高光

# 柱身上的金只出现在最上面这一圈（归一化高度），由顶面的糖霜往下淌出来。
# 这两个值决定"金压得多低"：0.58 起、到 1.0 满，也就是最上面四成里
# 由淡到浓地铺开，越往下越淡——再低就变成上下均匀的一圈黄带，
# 柱身会被拦腰截成两段。
GOLD_BAND_START = 0.58
GOLD_BAND_SPAN = 0.42

# 单颗透明度。有了深度测试（GL_LESS），每个像素最终只由**一颗粒子**决定，
# 所以这里给的就是"这个像素有多亮"，直接按目标亮度写就行，
# 不再需要为了"叠几十层不过曝"而压得极低。
# 区间也要窄：自转时同一像素上"最前面那颗"会换人，区间宽 = 换人时亮度跳变 = 闪。
CAKE_SIDE_ALPHA = (0.70, 0.76)       # 侧面：一层的亮度，正好是宝蓝该有的浓度。
                                     # 上限 0.76 是个卡点：侧面最亮的通道（蓝，0.60）
                                     # 乘上渐变的顶部系数 1.12 再乘 0.76 是 0.51，
                                     # 刚好压在 bloom 阈值 0.55 下面。再抬蛋糕体就自己发光。
                                     # 区间要**窄**：一个像素最终只显示一颗粒子，
                                     # 区间多宽，静止时表面的亮度噪声就有多大。
                                     # 原先 0.58~0.72（21% 的落差）就是柱身发麻的根源
CAKE_TOP_ALPHA = (0.66, 0.75)        # 顶面：参考图里最亮的一片，但仍然**不过辉光阈值**。
                                     # 原先给 0.90~1.00，理由是"它是整张图的光源"——
                                     # 但顶面是画面里最大的一块亮区，一过阈值，
                                     # 近层辉光就在它下面糊出一条二十多像素宽的灰白带，
                                     # 柱身上段那块宝蓝被洗掉，像蒙了层雾。
                                     # 这个值是按参考图反推的：要让合成结果落在
                                     # 中位数 0.784（见 COL_CREAM 的说明），
                                     # 解 0.94·a + 0.42·(1.2a−0.55) + 0.18·(…) = 0.784
                                     # 得 a ≈ 0.67。降到这个数之后顶面依然比柱身亮一倍，
                                     # 但不再是"自己烧起来"的一团，漩涡纹也才显出来 ——
                                     # 之前整片压在 1.0 上，金漩涡和奶白底全被削平成一个色
# （原先这里还有个 CAKE_INNER_ALPHA，是"内部垫体积"那批粒子用的。
#   内部粒子已经删掉了（见 make_cake_particles 的说明），常量一并删掉 ——
#   留着一个没人用的常量，下次读代码的人会以为还有一层内部粒子。）


# 底盘：现在盘子画在蛋糕**之后**并且参与深度测试，它落在蛋糕后面那半圈
# 会被深度缓冲正确判掉，不会再透过蛋糕叠上来 —— 所以这里可以正常给亮度，
# 盘子才看得出是一块浅薰衣草白的圆盘，而不是一圈若有若无的白点。
PLATE_ALPHA = (0.60, 0.70)

# ------------------------------ 彩色纸屑 ------------------------------
# 参考图里蛋糕周围撒着一层大小不一的彩点，左上角尤其密
CONFETTI_ALPHA = (0.30, 0.68)
                                     # 上限别给到 1.0：纸屑走的是加法，贴图又是柔光球，
                                     # alpha 一过 0.8，十几像素的球心就顶到纯白，
                                     # 落在蛋糕上的那几颗把顶面糊成一团白 ——
                                     # 参考图里压在蛋糕上的光斑是**带颜色**的（蓝的、金的），
                                     # 能透过它看见底下的蛋糕，不是一块白
CONFETTI_COLORS = (
    (0.30, 0.60, 1.00),   # 蓝
    (0.40, 0.90, 1.00),   # 青
    (1.00, 0.80, 0.30),   # 金
    (1.00, 0.60, 0.20),   # 橙
    (1.00, 0.55, 0.82),   # 粉
    (0.95, 0.96, 1.00),   # 白
)
CONFETTI_SPRAY = (-1.55, 0.95, 0.6)  # 密集簇的中心 (x, y, z) —— 参考图里在左上
CONFETTI_SPRAY_R = 1.45              # 簇的半径
CONFETTI_SPRAY_RATIO = 0.28          # 多大比例的纸屑集中在左上那一簇里。
                                     # 原先给 0.72 —— 七成的彩点全压在一个球里，
                                     # 屏幕上就是"只有左上角一块有粒子效果"，
                                     # 其余三面干干净净。现在反过来：只留不到三成做方向感，
                                     # 剩下七成绕着蛋糕铺一圈，四个方向都看得到飘着的彩点。
SPARK_COLOR = (1.00, 0.86, 0.42)     # 四角星闪光：金色
SPARK_ALPHA = (0.55, 0.95)

# ------------------------------ 光环 ------------------------------
# 对照 1.png：两圈倾斜的细光环 —— 上圈绕在蜡烛那一带，下圈绕在蛋糕下段。
# 每圈用两股半径略有差别的粒子叠出来，看起来才是"双线"，而不是一根粗管子。
RING_WIDTH = 0.0085                  # 环的粗细（半径方向的标准差）—— 这是**基准**，
                                     # 实际每一段还要乘上提按和细波（见 make_ring_particles），
                                     # 真正落地是 σ 从 0.0007 到 0.0120 之间变。
                                     # 从 0.0045 提到 0.0085 是因为量了一下：
                                     # 环半径 1.42，投影到 1000px 宽的画面上
                                     # 大约是 158 像素/单位，0.0045 的 σ 只有 0.95 像素厚 ——
                                     # 比粒子自己的柔光贴图（2~4 像素）还小。
                                     # 也就是说环的厚度**完全由贴图决定**，几何上
                                     # 再怎么提按，屏幕上都看不出来，只剩亮度在变。
                                     # 参考图里的飘带是一段真的有 3~5 像素宽的亮带，
                                     # 得让粗的那头几何厚度超过贴图，粗细才看得见。
                                     # 上限压在 0.0120（≈ ±0.036）：再粗就成宽带子了，
                                     # 参考图的环最粗也就是一根带子，不是色块
# 环走**发光**那一套（柔光球贴图 + 加法混合），不走实体那一套。
# 之前它是实体（方点 + 覆盖式写），结果是：每个像素只有一颗粒子说了算，
# 环就成了"一条 4 像素宽的实心白线"，边上没有任何过渡 —— 看起来是白胶带。
# 参考图里的环是一条**发光的**线：中间烧到纯白，往外一圈圈暗下去，
# 再外面是辉光。要这个剖面就必须让一个像素上叠十几颗粒子、亮度累加起来，
# 也就是加法混合。叠出来的亮度再交给 bloom，光晕才是从环里透出来的。
# 每颗粒子的透明度因此必须**很低**：一像素上叠十几层，
# 单颗 0.1 就已经是 1.0 出头，再高就只有纯白、颜色全丢了。
# 四条**轨道面**，每条一个倾角。倾角必须**不一样** —— 参考图里绕着蛋糕的
# 不只是两圈平行环，而是几条互为交叉的弧线，交叉点才是"轨道"的观感来源。
# 原先两圈都挂在一个 RING_TILT=22 的 GL 旋转上，是两条平行环，
# 看起来只是"两个圈"，不像在绕着蛋糕转。
# 后两条（-28° / 48°）更暗更细：它们是衬托，抢戏了就成了一圈圈箍。
#
# 倾角直接**烘进粒子坐标**（绕 X 轴转），不再靠绘制时的 glRotatef ——
# 一个 GL 旋转只能给一个角度，烘进坐标才能一条一个倾角。
#
# 每个轨道面 = (倾角, 中心高度, 该面占全部环粒子的比例, 面内各股)
# 每股 = (半径, 颜色, 透明度区间, 该股在面内的占比)
RING_PLANES = (
    (22.0,  0.72, 0.28, (          # 上圈：冷白，参考图里最亮的一条
        (1.420, (0.90, 0.94, 1.00), (0.29, 0.41), 0.62),
        (1.438, (0.62, 0.74, 1.00), (0.13, 0.20), 0.38),
    )),
    # 下圈是暖金，透明度要比上圈更收着：金的红通道本来就接近 1，
    # 叠到 1.5 倍就三条通道一起顶满、变成和白环一样的白线，金色就没了。
    # 峰值压在 1.0 上下，才既是"亮过阈值、会发光"，又还看得出是金的。
    # 但它也不能太弱：金的蓝通道只有 0.34，三通道加起来的总光量本来就比
    # 白环少一半，再压就只剩一条隐约的暖线，参考图里那圈金环是有存在感的。
    (22.0, -0.42, 0.26, (          # 下圈：暖金
        (1.372, (1.00, 0.80, 0.34), (0.30, 0.45), 0.58),
        (1.354, (0.80, 0.84, 1.00), (0.13, 0.20), 0.42),
    )),
    # 衬托用的两条：倾角取 -20° / 42°，和上面两圈交叉。半径和高度要一起收 ——
    # 一条半径 R、倾角 t 的环，竖直方向要甩出 R*sin(t)，半径一大或者倾角一陡，
    # 它的远侧就会从蛋糕顶面上方扫过去。顶面是整张图最亮的地方，
    # 一条线横在上面就是纯粹的噪点。这两条的控制目标就是"擦着柱身走"。
    # 颜色从冷蓝改成粉紫：上圈白、下圈金，四条里有三条都偏蓝，
    # 屏幕上看过去是一堆同色的环。参考图里交叉的那条飘带
    # 亮的地方是发粉的，不是又一个蓝环 —— 粉是这堆冷色里唯一的暖调对比。
    (-20.0, 0.12, 0.24, (          # 第三条：反着倾，粉紫
        (1.520, (1.00, 0.74, 0.92), (0.15, 0.22), 1.00),
    )),
    (42.0, -0.14, 0.22, (          # 第四条：更陡、更暗
        (1.480, (0.96, 0.86, 0.62), (0.10, 0.15), 1.00),
    )),
)

# ------------------------------ 蜡烛 ------------------------------
CANDLE_POS = (0.0, CAKE_TOP_Y, 0.0)  # 立在蛋糕顶面中心
CANDLE_R = 0.055
CANDLE_H = 0.62
CANDLE_COLOR = (1.00, 0.62, 0.86)    # 粉色（对照 1.png 的蜡烛）。
                                     # 别给到 0.52/0.80：那是荧光粉，在整根柱子上很扎眼
CANDLE_STRIPE = (1.00, 0.92, 0.98)   # 浅粉竖条纹（蜡烛的底色）
CANDLE_ALPHA = (0.62, 0.95)

# ------------------------------ 火焰 ------------------------------
FLAME_POS = (0.0, CAKE_TOP_Y + CANDLE_H, 0.0)   # 蜡烛顶部
FLAME_HEIGHT = 0.68
FLAME_BASE_R = 0.18
FLAME_COL_BOTTOM = (1.00, 0.98, 0.72)  # 底部 近白的亮黄（焰心）
FLAME_COL_MID = (1.00, 0.72, 0.16)     # 中部 橙
FLAME_COL_TOP = (1.00, 0.30, 0.02)     # 顶部 红
FLAME_RISE = (0.7, 1.5)              # 每颗粒子上浮速度（life/秒）
FLAME_JITTER = 0.020                 # 抖动幅度
FLAME_SWIRL = 1.2                    # 绕轴旋转速度
FLAME_DRIFT = 0.02                   # 越往上越发散
FLAME_TAPER = 1.10                   # 顶部收窄指数（越大越尖）
FLAME_ALPHA = (0.22, 0.48)           # 单颗透明度（粒子多，必须压暗才看得出颗粒）

# ------------------------------ 背景星星 ------------------------------
STAR_RADIUS = (8.0, 15.0)            # 球壳半径范围
STAR_ALPHA = (0.25, 0.80)
STAR_TWINKLE = (0.9, 2.4)            # 闪烁频率范围

# ------------------------------ 动画 ------------------------------
ROT_SPEED = 18.0                     # 蛋糕自转速度（度/秒）
# 环的"飘"—— 每帧重算几何用的三个参数。
# 这一套是"灵动"的关键：以前环的浮动走 ParticleSystem 那套**每颗粒子随机频率**
# 的上下抖，62000 颗粒子各抖各的，屏幕上只是一层毛边，动起来像在嗡嗡震，
# 不像一条带子在飘。现在整条环同一时刻只有一个相位，波峰沿弧长以固定速度跑。
RING_WAVE_AMP = 0.075                # 上下起伏幅度。蛋糕半径是 1.6，这是 4.7% ——
                                     # 再大环就开始"波浪"了，看得出手腕在画波浪线；
                                     # 再小就跟没动一样。0.045 的静态起伏之上再叠这个。
RING_WAVE_SPEED = 1.9                # 波沿弧长跑的速度（弧度/秒）。一圈 2π，
                                     # 也就是三秒多跑完一整圈 —— 肉眼看得出在流动，
                                     # 又不至于快到像弹簧在抖
RING_BREATH = 0.022                  # 半径呼吸幅度（±2.2%）
RING_BREATH_SPEED = 1.3              # 呼吸速度。跟上下波取不同的速度，
                                     # 两条运动不锁相，环就不会"整块平移"而是"扭"

RING_SPIN = -14.0                    # 光环自己的自转速度（度/秒）—— **负号就是反向**。
                                     # 环从蛋糕那一组里拆出来了：原先它跟蛋糕吃同一个
                                     # glRotatef(angle)，两套东西像焊在一起的一整块在转，
                                     # 看不出是「环在绕着蛋糕走」。反着来才有相对运动。
                                     # 数值取 14 而不是 -18：正好镜像的话两边速度一样，
                                     # 盯久了像同一台机器的两个齿轮；差一点才像各自在漂。
FLOAT_AMP = 0.03                     # 上下浮动幅度
FLOAT_FREQ = (0.5, 2.4)              # 浮动频率（每颗粒子随机）

# ------------------------------ 星云背景 ------------------------------
NEBULA_RADIUS = (3.0, 11.0)          # 星云雾分布范围（包住蛋糕）
NEBULA_ALPHA = (0.025, 0.055)        # 低透明度。调高就会变成一团团看得见的圆斑
NEBULA_COLORS = ((0.55, 0.20, 1.0), (0.15, 0.30, 1.0), (0.70, 0.20, 0.95))

# ------------------------------ 辉光 Bloom ------------------------------
# 方案 A：FBO + 高斯模糊
#   粒子 → FBO(scene) → 亮度提取 → 横向模糊 → 纵向模糊 → 与原图加法合成
#   画面 = 原图 × 0.5 + 模糊图 × 0.5
#
# 两个关键点，缺一个就会"看起来没有辉光"：
#   1. 模糊半径要**足够宽**。光晕是相对粒子尺寸而言的：粒子只有几像素，
#      如果模糊也只跨几像素，光晕就只是一圈贴边的毛边，整体仍是一盘彩沙。
#      这里的数值是 sigma，核实际跨越 ±3σ。
#   2. 高斯核是归一化的，亮点模糊后能量会被摊薄，必须给增益，否则光晕弱到看不见。
BLOOM_ENABLED = True
# 阈值决定"谁配发光"。调低会把整个蛋糕体都算进光晕里，深蓝的大理石面
# 被一层灰白的光雾盖住，就成了"蒙了一层灰" —— 只有火焰、光环、顶面高光
# 这些真正亮的东西该过阈值。
BLOOM_THRESHOLD = 0.55               # 只有亮过这个值的像素才参与光晕。
                                     # 别调低：顶层那片暖金是画面里最大的亮块，
                                     # 阈值一低它整个进光晕，远层模糊把它摊到整个
                                     # 蛋糕上 —— 红绿通道被抬起来，宝蓝就洗成了
                                     # 淡紫（实测蓝 R 通道从 0.11 涨到 0.26）
BLOOM_GAIN = 1.20                    # 亮度提取的增益。别调太高：太高会把亮度提成
                                     # 近似二值（超过阈值的一律顶到 1.0），光晕边缘会发硬
BLOOM_SIGMA_NEAR = 3.0               # 近层模糊 sigma（贴边的亮芯）
BLOOM_SIGMA_FAR = 6.5                # 远层模糊 sigma（在半分辨率上算，等效全分辨率 13）

# 权重：原图接近 1，辉光是**叠加**上去的。
# 如果按 原图0.5 + 模糊0.5 来合成，背景和暗部会被直接砍掉一半，整幅图变暗，
# 光晕补不回来 —— 实测开辉光比关辉光还暗（21.2 vs 27.4），那就不是"发光"了。
BLOOM_W_ORIGINAL = 0.94              # 原图权重（保持画面本来的亮度）
BLOOM_W_NEAR = 0.33                  # 近层模糊权重（贴着亮点的光边）。
                                     # 这一层可以给得比远层大得多：sigma 只有 3，
                                     # 光只贴在亮点周围几个像素，不会跑到蛋糕身上去。
                                     # 参考图里那种"什么都带一圈光"的观感主要靠它。
                                     # 远层（sigma 6.5）不行：摊得太开，权重一大
                                     # 就是给整张图蒙一层白雾，蓝色最先被冲掉。
BLOOM_W_FAR = 0.18                   # 远层模糊权重（大范围的光晕溢出）。
                                     # 之前只能给 0.10：那时**蛋糕体自己也亮过了阈值**，
                                     # 远层把顶面那片暖金抹遍整个柱身，宝蓝就洗成了淡紫。
                                     # 现在柱身压在阈值以下，过线的只有顶面 / 环 / 火焰 / 纸屑，
                                     # 远层溢出来的就只剩它们的光 —— 这正是参考图里
                                     # 蛋糕周围那一圈背景被照亮的效果

# 边缘粒子加强：越靠蛋糕外侧的粒子越大越亮，中心稍暗（星云包裹感）
EDGE_SIZE_BOOST = 0.45               # 最外侧粒子的尺寸加成
EDGE_ALPHA_BOOST = 0.14              # 最外侧粒子的亮度加成。别调高：上层顶面圆盘
                                     # 的外沿半径最大，会吃满这个加成，俯视时整个
                                     # 顶盘叠成一条带，直接过曝成白块

# ------------------------------ 相机 ------------------------------
CAM_DIST_START = 6.6
CAM_DIST_MIN, CAM_DIST_MAX = 3.0, 20.0
CAM_PITCH_START = 19.0               # 略俯视，顶面那片奶油金才看得见（对照 1.png）
CAM_PITCH_MIN, CAM_PITCH_MAX = -85.0, 85.0
DRAG_SENS = 0.32
ZOOM_SENS = 0.6
FOV = 45.0
NEAR, FAR = 0.1, 100.0

# ------------------------------ 文字 ------------------------------
FONT_NAME = "arial,segoeui,helvetica,dejavusans"
TEXT_MAIN = "Happy Birthday"
TEXT_MAIN_SIZE = 64
TEXT_MARGIN_BOTTOM = 42
TEXT_COLOR = (235, 240, 255)
TEXT_GLOW_RGB = (90, 130, 220)

SEED = None                          # 填整数可固定随机效果


# ==============================================================================
# ② 通用工具
# ==============================================================================

rng = np.random.default_rng(SEED)


def _gl_id(handle):
    """兼容 glGen* 返回标量或数组"""
    return int(np.asarray(handle).ravel()[0])


def random_sizes(n, size_range):
    """
    每颗粒子在给定区间里随机取大小，**取整**分档。
    取整很重要：ParticleSystem.draw() 是按大小合批的，如果每颗粒子都是
    各不相同的浮点大小，就会退化成几千次 draw call。
    """
    return np.rint(rng.uniform(size_range[0], size_range[1], n)).astype(np.float32)


def gaussian_weights(sigma, sigmas_out=3.0):
    """
    生成一维高斯核，返回 (weights, half)。
    sigma 是标准差（像素），核覆盖 ±sigmas_out*sigma 的范围。
    注意：一半的"辉光看不见"都是因为把这里当成"半径"用了 —— 半径 6 如果
    当成 sigma=3、只跨 ±6 像素，相对于几百像素宽的蛋糕就只是一圈贴边毛刺。
    归一化保证整体亮度不变，但也会摊薄亮点能量，所以外面还要有增益补偿。
    """
    half = max(1, int(math.ceil(sigma * sigmas_out)))
    offs = np.arange(-half, half + 1, dtype=np.float64)
    w = np.exp(-(offs ** 2) / (2.0 * sigma * sigma))
    w /= w.sum()
    return w.astype(np.float32), half


# ==============================================================================
# ③ 粒子生成函数（纯 numpy，dtype 全部 float32）
# ==============================================================================

def _cylinder_side(n, radius, y_bottom, height, y_jitter=0.006):
    """圆柱侧面撒点，返回 (pos, theta, y_normalized)。
    把 theta 和归一化高度一起返回，是为了让颜色能按位置算大理石纹。"""
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    yy = rng.uniform(y_bottom, y_bottom + height, n)
    rr = radius * (1.0 + rng.normal(0.0, 0.006, n))
    pos = np.stack([rr * np.cos(th), yy, rr * np.sin(th)], axis=1)
    return pos, th, (yy - y_bottom) / max(height, 1e-6)


def _disc(n, radius, y_plane, y_jitter=0.006):
    """水平圆盘撒点（按面积均匀，所以半径取 sqrt），返回 (pos, 归一化半径, theta)"""
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    u = np.sqrt(rng.uniform(0.0, 1.0, n))
    rr = radius * u
    yy = y_plane + rng.normal(0.0, y_jitter, n)
    pos = np.stack([rr * np.cos(th), yy, rr * np.sin(th)], axis=1)
    return pos, u, th


def _marble_rgb(th, yn):
    """
    侧面的花纹：宝蓝底子上洇着一团团深藏青/亮蓝的云斑，
    最上段再压一圈被 θ 掐断的金 —— 对应 1.png 里那种
    "水彩蓝的柱身 + 顶上淌下来的糖霜金"。
    注意这是按**位置**算的，不是随机撒色，所以旋转时纹路会跟着一起转，
    看起来才是长在蛋糕上的纹理。
    """
    # --- 蓝：一团团的斑，不是一道道纹 ---
    # 放大 1.png 看柱身：底色是宝蓝，上面洇着一块块边界模糊的深藏青暗斑，
    # 像水彩——斑的尺度有柱身宽度的三分之一，是**圆**的。
    # 之前一直在调"金脉"，把 θ 频率拉到 34、高度压到 6.4π，格子拉成 1:4 的
    # 长条，出来是几道笔直的黄划痕 —— 那是把参考图读错了：
    # 柱身上根本没有长条纹，金只是顶上薄薄一圈 + 撒落的圆点。
    # 现在三个低频正弦叠加，θ 和高度**同量级**（2.0~3.7 对 1.1~3.1π），
    # 一个格子接近正方，出来的才是圆斑。
    # 三个频率而不是两个：两个叠出来的波峰在 θ 上有明显周期，
    # 绕一圈能看到花纹重复三四次，一眼就是"程序生成"。
    cloud = (0.50 * np.sin(2.0 * th + 1.1 * yn * math.pi + 0.4)
             + 0.32 * np.sin(3.7 * th - 1.9 * yn * math.pi + 2.3)
             + 0.18 * np.sin(1.3 * th + 3.1 * yn * math.pi - 1.2))
    # 指数 >1 把亮蓝压回少数：宝蓝是主调，亮蓝只在云斑的亮面上透出来。
    # 取 1.0 的话两种蓝各占一半，整体偏成浅蓝，参考图里那块深蓝的底子就没了。
    # 但也不能太狠（试过 2.8）：亮蓝被压没了，柱身就是一大片均匀的深蓝。
    blue_t = np.clip(0.5 + 0.80 * cloud, 0.0, 1.0) ** 1.4

    # --- 金：顶上淌下来的一圈，被 θ 打断 ---
    # 参考图的金集中在柱身最上面四分之一 —— 顶面那层糖霜往下淌出来的，
    # 底部几乎看不到金。之前是"哪个波峰过线哪里就转金"，金从上到下均匀分布，
    # 柱身被切成几条黄色斜带；而金色**只该出现在上段**这一条，
    # 比"金脉长什么样"重要得多。
    band = np.clip((yn - GOLD_BAND_START) / GOLD_BAND_SPAN, 0.0, 1.0) ** 1.5
    # 匀匀一圈会看成"蛋糕分层"，参考图里这圈是断断续续淌下来的：
    # 用 θ 上的一个慢波把它掐出粗细。
    drip = 0.5 + 0.5 * np.sin(2.6 * th + 1.7 * yn * math.pi + 0.9)
    # 系数 0.72 是个卡点，跟 CAKE_SIDE_ALPHA 的 0.76 是同一类计算：
    # 最亮的那颗金（红 0.90）乘竖直渐变的顶部系数 1.12 再乘 0.72 是 0.73，
    # 乘上 alpha 0.76 落到 0.55 —— 正好压在 bloom 阈值上。
    # 给到 1.0 的话这一圈金整个亮过阈值，辉光一叠，柱身上段会糊成一片
    # 灰白的雾（金色的红通道太高，辉光是按通道加的，红先冲顶）。
    gold_t = np.clip(band * (0.20 + 1.25 * drip), 0.0, 1.0) * 0.72

    royal = np.asarray(COL_ROYAL_BLUE, dtype=np.float32)
    mid = np.asarray(COL_MID_BLUE, dtype=np.float32)
    gold = np.asarray(COL_GOLD_LIGHT, dtype=np.float32)
    base = royal * (1.0 - blue_t[:, None]) + mid * blue_t[:, None]
    rgb = base * (1.0 - gold_t[:, None]) + gold * gold_t[:, None]
    # 这里**不再**给金色额外的亮度加成。金色之所以看起来像刷上去的油漆，
    # 就是因为它亮过了 bloom 阈值：辉光往上叠一层，红绿一起抬，
    # 琥珀色就冲成了柠檬黄白。金脉只要靠 alpha 稍微亮一点就够 ——
    # 颜色本身是琥珀（1.00/0.76/0.28），压在阈值以下才是石头上的纹，
    # 亮过阈值就成了荧光棒。
    # 上亮下暗的竖直渐变：顶面是整张图的光源，柱身上段被它照着，
    # 下段落进盘子的阴影里。少了这一层，柱身就是一片均匀的蓝，
    # 看不出是个圆柱，更像一张贴了花纹的纸。
    # 范围 0.66~1.06：底部压掉三分之一，顶部微微提亮。
    # 上限不能再高 —— 顶部最亮的通道（蓝 0.60）乘 1.06 再乘 alpha 0.80
    # 就是 0.51，离 bloom 阈值 0.55 只剩一点余量，再抬蛋糕体就自己发光了。
    rgb = rgb * (0.52 + 0.60 * yn)[:, None]

    # 每颗粒子的随机明暗差要**很小**。每个像素最终只取最前面那一颗，
    # 自转时"中选"的粒子不停换人；每颗之间差得多大，表面就闪得多厉害。
    rgb = rgb * rng.uniform(0.95, 1.0, len(th)).astype(np.float32)[:, None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)



def make_cake_particles(total=CAKE_COUNT):
    """
    蛋糕主体：单层矮圆柱（对照 1.png）。
      · 侧面 —— 宝蓝底 + 金色/亮蓝大理石漩涡（按位置算，跟着蛋糕一起转）
      · 顶面 —— 暖奶油金的发光面，参考图里最亮的一片
      · 内部 —— 很淡的粒子，只用来垫出体积
    """
    n_surf = int(total * CAKE_SURFACE_RATIO)

    # 表面粒子按"侧面面积 : 顶面面积"分配，但顶面再打个折：
    # 顶面是平铺的一片，粒子在屏幕上叠得极狠（实测叠加近 9 层），
    # 按面积平分的话必然过曝成一块白。密度降下来，靠大颗粒铺满就行。
    side_area = 2.0 * math.pi * CAKE_R * CAKE_H
    top_area = math.pi * CAKE_R * CAKE_R
    n_top = int(n_surf * top_area / (side_area + top_area) * CAKE_TOP_DENSITY)
    n_side = n_surf - n_top

    pos_list, col_list, size_list = [], [], []

    # --- 侧面：大理石纹 ---
    pos, th, yn = _cylinder_side(n_side, CAKE_R, CAKE_Y, CAKE_H)
    rgb = _marble_rgb(th, yn)
    a = rng.uniform(CAKE_SIDE_ALPHA[0], CAKE_SIDE_ALPHA[1], n_side).astype(np.float32)
    # 金色不再额外提亮。金色本来就比蓝亮一大截，再给 alpha 加成，
    # 屏幕上就是三条荧光带压在蛋糕上，把整个柱身都抢掉了 ——
    # 参考图里金只是蓝底子上的点缀，纹路的亮度仍由蓝色那一档说了算。

    # --- 石面上的闪点 ---
    # 参考图的蓝底子上撒着一层细小的亮点，像石头里嵌的云母。
    # 少了这一层，蛋糕体就是一大片均匀的哑光蓝，再干净的纹路也显得"死"。
    # 闪点是**彩色**的（暖白为主，掺金、淡蓝、粉），不是清一色白 ——
    # 参考图的柱身上那些细点明显带着颜色，一路看过去是星星点点的，
    # 全给白的话在蓝底上就是一层均匀的噪点，反而显得脏。
    n_speck = int(n_side * CAKE_SPECK_RATIO)
    idx = rng.choice(n_side, n_speck, replace=False)
    palette = np.asarray(SPECK_COLORS, dtype=np.float32)
    pick = rng.integers(0, palette.shape[0], n_speck)
    rgb[idx] = palette[pick] * rng.uniform(0.85, 1.0, n_speck).astype(np.float32)[:, None]
    # 闪点的透明度给到接近 1：实体是不透明覆盖式画的，
    # 一颗粒子就是一个像素，够亮才闪得起来
    a[idx] = rng.uniform(0.85, 1.0, n_speck).astype(np.float32)

    pos_list.append(pos)
    col_list.append(np.concatenate([rgb, a[:, None]], axis=1))
    size_list.append(random_sizes(n_side, CAKE_SIZE_RANGE))

    # --- 顶面：奶白底 + 几道金色漩涡 + 前沿一圈金 ---
    # 原先这里是"中心白热 → 中段奶油 → 外沿金"的同心插值，出来是一圈
    # 橙色甜甜圈：三段都绕中心对称，而参考图的顶面根本不是同心环 ——
    # 是奶白的糖霜上淌着几道走向不规则的金，金最厚的地方在靠前那一侧。
    pos, u, th_top = _disc(n_top, CAKE_R, CAKE_TOP_Y)
    cream = np.asarray(COL_CREAM, dtype=np.float32)
    rim = np.asarray(COL_CREAM_EDGE, dtype=np.float32)
    gold = np.asarray(COL_CREAM_GOLD, dtype=np.float32)
    # 两道低频波扫过整张顶面（一道沿 θ 走、一道斜着穿过半径），
    # 相位不同的地方就是金的漩涡。单一频率只会是一圈同心环。
    sw = (0.55 * np.sin(2.3 * th_top + 3.2 * u * math.pi + 0.7)
          + 0.45 * np.sin(3.9 * th_top - 5.1 * u * math.pi + 2.4))
    swirl = np.clip((sw + 0.10) / 0.40, 0.0, 1.0)
    mix = (swirl * 0.82)[:, None]
    rgb = cream * (1.0 - mix) + gold * mix
    # 最外沿收一道金：顶面的边是立起来的糖霜，光打在那圈上最暖。
    # 少了它，顶面和柱身之间会"断"成两截。
    e = np.clip((u - 0.82) / 0.18, 0.0, 1.0)[:, None]
    rgb = rgb * (1.0 - e) + rim * e
    rgb *= rng.uniform(0.95, 1.0, n_top).astype(np.float32)[:, None]
    a = rng.uniform(CAKE_TOP_ALPHA[0], CAKE_TOP_ALPHA[1], n_top).astype(np.float32)
    pos_list.append(pos)
    col_list.append(np.concatenate([rgb, a[:, None]], axis=1))
    size_list.append(random_sizes(n_top, CAKE_SIZE_RANGE))

    # 不再生成"内部垫体积"的粒子：实体面是不透明的，内部粒子永远看不见，
    # 唯一的后果是它们的深度和外壳只差 ±0.006，总有几个像素被它们抢到，
    # 表面就多出一层高频麻点（实测最明显的就是这一批）。
    # 省下来的粒子数全部给外壳，表面更密。
    return {"pos": np.concatenate(pos_list, axis=0).astype(np.float32),
            "col": np.concatenate(col_list, axis=0).astype(np.float32),
            "size": np.concatenate(size_list, axis=0).astype(np.float32),
            "float_amp": FLOAT_AMP}


def make_plate_particles(n=PLATE_COUNT):
    """
    底盘：一块比蛋糕大的扁圆盘（对照 1.png 底下那个浅色盘子）。
    盘面用低透明度的淡薰衣草白，边缘一圈加密加亮，做出盘沿。
    """
    # 盘沿只占一小撮。原先给 22% 的粒子：那 2600 颗全挤在半径 1.52 的一圈细线上，
    # 每颗又是 4~9 像素的方点，互相压得死死的 —— 屏幕上就是一圈死白的厚墙，
    # 盘子成了个白甜甜圈。参考图里的盘沿只是一道稍微亮一点的高光。
    n_face = int(n * 0.88)
    n_rim = n - n_face

    pos_list, col_list, size_list = [], [], []

    # --- 盘面 ---
    pos, u, _ = _disc(n_face, PLATE_R * 0.93, PLATE_Y, y_jitter=0.004)
    plate = np.asarray(COL_PLATE, dtype=np.float32)
    rgb = plate * rng.uniform(0.92, 1.0, n_face).astype(np.float32)[:, None]
    # 盘面基本是平的，只带一点点外沿积光。
    # 原先给的是 (1.10 + 0.6*u)，从 1.10 一路涨到 1.70 —— 那不是"积光"，
    # 是在盘子上又画了一个亮环：中心暗、外圈亮，屏幕上仍然是个甜甜圈，
    # 只是颜色从白换成了灰紫。参考图里的盘子是**均匀**的一块，
    # 亮度靠整块给够，不是靠给边缘加码。
    a = rng.uniform(PLATE_ALPHA[0], PLATE_ALPHA[1], n_face).astype(np.float32)
    a = np.clip(a * (0.92 + 0.16 * u), 0.0, 1.0)
    pos_list.append(pos)
    col_list.append(np.concatenate([rgb, a[:, None]], axis=1))
    size_list.append(random_sizes(n_face, PLATE_SIZE_RANGE))

    # --- 盘沿：半径收窄的一圈，并且微微翘起 ---
    th = rng.uniform(0.0, 2.0 * math.pi, n_rim)
    rr = PLATE_R * (1.0 + rng.normal(0.0, 0.012, n_rim))
    yy = PLATE_Y + PLATE_RIM_H * rng.uniform(0.0, 1.0, n_rim) ** 2
    pos = np.stack([rr * np.cos(th), yy, rr * np.sin(th)], axis=1)
    edge = np.asarray(COL_PLATE_EDGE, dtype=np.float32)
    rgb = edge * rng.uniform(0.85, 1.0, n_rim).astype(np.float32)[:, None]
    # 盘沿只是比盘面亮一档，不是另一块白。给到 2.0 倍（也就是 1.0）就是纯白。
    a = rng.uniform(PLATE_ALPHA[1] * 1.05, PLATE_ALPHA[1] * 1.45,
                    n_rim).astype(np.float32)
    pos_list.append(pos)
    col_list.append(np.concatenate([rgb, a[:, None]], axis=1))
    size_list.append(random_sizes(n_rim, PLATE_SIZE_RANGE))

    return {"pos": np.concatenate(pos_list, axis=0).astype(np.float32),
            "col": np.concatenate(col_list, axis=0).astype(np.float32),
            "size": np.concatenate(size_list, axis=0).astype(np.float32),
            "float_amp": 0.0}



def make_ring_particles(n=RING_COUNT):
    """
    光环：对照 1.png，绕着蛋糕的几条倾斜弧带（见 RING_PLANES 的说明）。
    每个轨道面里的每一股各自撒一圈粒子，绕 X 轴按该面的倾角转过去。

    关键是**不能画成等宽等亮的正圆**。参考图里的环是手绘的飘带：
    宽窄一路在变，亮暗也在换。等宽等亮的正圆在屏幕上就是一个铁圈。

    让环动起来靠三件事，前两件烘在几何里，第三件每帧算：

      1. **宽窄沿弧长变**（提按）—— 等宽的圈是个铁箍，宽窄有起伏才像笔画的。
      2. **不在一个平面上** —— 环上下有静态的低频起伏，看着就不是个正圆。
      3. **起伏要一路跑起来** —— 这是"灵动"的真正来源，见 ring_animate()。

    第 3 件要求把几何**参数**（每颗粒子的 θ、所在面的倾角、基准半径/高度）
    一并带出去（"anim" 里的那几项），每帧重新算位置。
    """
    n_plane = np.asarray([p[2] for p in RING_PLANES], dtype=np.float64)
    n_plane = np.floor(n_plane / n_plane.sum() * n).astype(int)
    n_plane[-1] = n - n_plane[:-1].sum()

    pos_list, col_list, size_list = [], [], []
    th_list, ct_list, st_list = [], [], []
    r_list, y_list, pv_list, pr_list, am_list = [], [], [], [], []

    for pi, ((tilt, y0, _, strands), pn) in enumerate(zip(RING_PLANES, n_plane)):
        w = np.asarray([s[3] for s in strands], dtype=np.float64)
        counts = np.floor(w / w.sum() * pn).astype(int)
        counts[-1] = pn - counts[:-1].sum()

        t = math.radians(tilt)
        ct, st = math.cos(t), math.sin(t)

        # 每条轨道面一组自己的相位。四条环的起伏如果在同一处收，
        # 屏幕上就是四条同步呼吸的圈 —— 那种规律感正是"僵硬"的来源。
        ph = rng.uniform(0.0, 2.0 * math.pi, 4)
        # 走得快的环幅度也大一点。四条如果摆得一样大，看过去是四块一起晃；
        # 幅度错开，才是各自在飘。
        amp = RING_WAVE_AMP * (0.62 + 0.55 * abs(math.sin(ph[0])))

        for si, ((radius, rgb, arange, _), cnt) in enumerate(zip(strands, counts)):
            th = rng.uniform(0.0, 2.0 * math.pi, cnt)

            # --- 1. 粗细沿弧长变：毛笔的提按 ---
            # 两道波叠：2.1 是主提按，3.6 是第二道细的，两个频率不成整数倍，
            # 所以收笔的位置不会每圈重复在同一处。
            # **都不夹到零**。上一版这里是 (x-0.18)/0.42 的硬夹，实测把 29%
            # 的粒子压到了下限 —— 环上就出现一段段几乎没粒子的细丝，
            # 加法叠不出亮度，看着是断断续续的绳子。环可以细、可以暗，
            # 但不该断；断续交给 alpha 那道波去做，它断得起。
            taper = (0.5 + 0.5 * np.sin(2.1 * th + ph[0])) ** 1.5
            thin = 0.5 + 0.5 * np.sin(3.6 * th + ph[1])
            # 两头都拉开：最细 0.08×RING_WIDTH（亚像素，会真的淡掉），
            # 最粗 1.41×（约 5~6 像素的亮带）。
            # 参考图里那两条环就是一头宽亮、绕到后面细到几乎没有 ——
            # 环的"流向"一半来自 alpha 那道波，另一半就来自这个粗细差。
            width = RING_WIDTH * (0.18 + 1.05 * taper) * (0.45 + 0.70 * thin)

            # --- 2. 不是正圆：半径和高度都带低频起伏 ---
            # 半径 ±2% 就够：再多环就"瘪"了，看得出来是个歪圈而不是手绘感
            rr = radius * (1.0 + 0.020 * np.sin(3.0 * th + ph[2])) \
                + rng.normal(0.0, 1.0, cnt) * width
            # 上下 ±0.045：环不在一个平面上，是飘着的。
            # 这个幅度压在蛋糕半径的 4% 以内，肉眼读成"手抖"而不是"波浪"
            yy = y0 + 0.045 * np.sin(2.0 * th + ph[3]) \
                + rng.normal(0.0, RING_WIDTH * 0.7, cnt)

            th_list.append(th.astype(np.float32))
            ct_list.append(np.full(cnt, ct, np.float32))
            st_list.append(np.full(cnt, st, np.float32))
            r_list.append(rr.astype(np.float32))
            y_list.append(yy.astype(np.float32))
            # 每颗粒子带上自己那条环的相位。每帧重算时同一条环上的粒子
            # 拿到的是同一个相位，跑起来的波才是整条带子在飘，
            # 而不是 62000 个各抖各的（那就是一层毛边）。
            # +pi*1.7 让四条环的波错开，不至于一起到波峰。
            pv_list.append(np.full(cnt, ph[3] + pi * 1.7, np.float32))
            pr_list.append(np.full(cnt, ph[2] + pi * 0.9, np.float32))
            am_list.append(np.full(cnt, amp, np.float32))

            base = np.tile(np.asarray(rgb, dtype=np.float32), (cnt, 1))
            base *= rng.uniform(0.92, 1.0, cnt).astype(np.float32)[:, None]

            # --- 3. 亮暗沿弧长变，而且**两股错开** ---
            # 每股的相位差 2.1 弧度：同一段弧上，一股正亮、另一股正暗。
            # 于是一圈看过去，颜色在"冷白"和"淡蓝"（或"金"和"淡蓝"）之间来回换 ——
            # 参考图里的环就是这么一段偏白、一段偏蓝的，不是一圈一个颜色。
            # 系数 0.06 + 1.75：整圈的平均倍率仍在 0.73 上下（跟原来差不多），
            # 但**谷底从 0.30 压到了 0.06**，峰从 1.70 抬到 1.81。
            # 这一项是"灵动"里最管用的一处。原先谷底 0.30 太高 ——
            # 一圈下来最暗的地方也还有三成亮度，屏幕上就是一条**整圈都亮**的
            # 白箍，只是亮暗略有不同。参考图里的飘带不是这样：它大部分是
            # 一条很淡的细线，只有一段烧成亮白，然后淡出去。
            # 谷底压到 0.06（加法叠完约等于没有），暗段才真的沉下去，
            # 那一段亮白才有"扫过去"的感觉。
            # 指数从 1.2 提到 1.6：把中间值往暗侧推，亮的部分更集中成一段。
            wave = 0.5 + 0.5 * np.sin(2.6 * th + ph[0] + si * 2.1)
            # 每颗粒子的基础透明度**不能取满整个区间**。
            # 原来直接 uniform(arange[0], arange[1])，区间宽的有 1.5 倍，
            # 于是暗段上总有一批粒子随机拿到高值，一颗颗凸出来 ——
            # 屏幕上就是"虚线"，不是"淡出"。淡出要的是整段一起暗下去，
            # 所以这里把散布收到区间的 ±30%：保留了颗粒感，但不再有离群的亮点。
            a_mid = 0.5 * (arange[0] + arange[1])
            a_half = 0.5 * (arange[1] - arange[0])
            a = (a_mid + rng.uniform(-0.30, 0.30, cnt) * a_half).astype(np.float32)
            a = a * (0.06 + 1.75 * wave ** 1.6).astype(np.float32)
            col_list.append(np.concatenate([base, a[:, None]], axis=1))
            size_list.append(random_sizes(cnt, RING_SIZE_RANGE))

            # 初始位置：静态那两道起伏就算进去了。
            # 绕 X 轴转 tilt：只动 y / z，x 不变
            zz = rr * np.sin(th)
            pos_list.append(np.stack([rr * np.cos(th),
                                      yy * ct - zz * st,
                                      yy * st + zz * ct], axis=1))

    th = np.concatenate(th_list)
    ct = np.concatenate(ct_list)
    st = np.concatenate(st_list)
    r = np.concatenate(r_list)
    y = np.concatenate(y_list)

    return {"pos": np.concatenate(pos_list, axis=0).astype(np.float32),
            "col": np.concatenate(col_list, axis=0).astype(np.float32),
            "size": np.concatenate(size_list, axis=0).astype(np.float32),
            # 环的浮动自己算（ring_animate），不走 ParticleSystem 那套
            # 每颗粒子随机频率的上下抖 —— 那套对环只会抖出一层毛边。
            "float_amp": 0.0,
            "anim": {
                "th": th, "ct": ct, "st": st, "r": r, "y": y,
                "cos_th": np.cos(th).astype(np.float32),
                "sin_th": np.sin(th).astype(np.float32),
                "ph_v": np.concatenate(pv_list),
                "ph_r": np.concatenate(pr_list),
                "amp": np.concatenate(am_list),
                # 预分配的输出缓冲：每帧往这里写，避免 62000×3 的反复分配
                "out": np.empty((len(th), 3), np.float32),
            }}


def ring_animate(sys_ring, t):
    """
    光环每帧的几何：两条行波。

      · 上下：y 上加一道 sin(2θ - ωt + φ)。波峰沿弧长以固定速度往前跑，
        所以看到的是"带子在飘"，不是"整块在动"。
      · 半径：轻微呼吸，速度和相位都跟上下波错开，两条运动不锁相，
        环就有了"扭"的感觉。

    静态那两道起伏（±2% 半径、±0.045 高度、粒子粗细的随机散布）
    已经烘在 r / y 里了，这里只往上叠**动的**部分 ——
    散布跟着动的话，环会变成一团抖动的噪点，反而更僵硬。
    """
    d = sys_ring.anim
    if d is None:
        return
    th = d["th"]
    cth, sth = d["cos_th"], d["sin_th"]

    yy = d["y"] + d["amp"] * np.sin(2.0 * th - RING_WAVE_SPEED * t + d["ph_v"])
    rr = d["r"] * (1.0 + RING_BREATH * np.sin(
        3.0 * th + RING_BREATH_SPEED * t + d["ph_r"]))

    zz = rr * sth
    out = d["out"]
    out[:, 0] = rr * cth
    out[:, 1] = yy * d["ct"] - zz * d["st"]
    out[:, 2] = yy * d["st"] + zz * d["ct"]
    sys_ring.update_positions(out)


def make_candle_particles(n=CANDLE_COUNT):
    """
    蜡烛：细圆柱，粉色带浅色竖条纹（对照 1.png 的粉色蜡烛）。
    条纹按 theta 取，所以是绕柱身的一圈圈竖纹，不是随机花色。
    """
    cx, cy, cz = CANDLE_POS
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    rr = CANDLE_R * (1.0 + rng.normal(0.0, 0.05, n))
    yy = cy + rng.uniform(0.0, CANDLE_H, n)

    pos = np.stack([cx + rr * np.cos(th), yy, cz + rr * np.sin(th)], axis=1)

    pink = np.asarray(CANDLE_COLOR, dtype=np.float32)
    stripe = np.asarray(CANDLE_STRIPE, dtype=np.float32)
    # 绕柱身 6 道条纹，取波峰附近为浅色
    # 阈值往下挪（-0.35 → -0.05）并且把过渡带拉宽：原来 t 大部分是 0，
    # 整根蜡烛几乎全是饱和的粉色。参考图里那根蜡烛是**浅色**为主、
    # 粉只是绕在上面的条纹，底色偏白。
    t = np.clip((np.sin(th * 6.0) - 0.05) / 0.75, 0.0, 1.0)[:, None]
    rgb = pink * (1.0 - t) + stripe * t
    rgb *= rng.uniform(0.85, 1.0, n).astype(np.float32)[:, None]
    a = rng.uniform(CANDLE_ALPHA[0], CANDLE_ALPHA[1], n).astype(np.float32)
    col = np.concatenate([rgb, a[:, None]], axis=1)
    return {"pos": pos.astype(np.float32), "col": col.astype(np.float32),
            "size": CANDLE_POINT_SIZE, "float_amp": 0.0}


def make_star_particles(n=STAR_COUNT):
    """
    背景星星：半径 8~15 的球壳，大小 1~2 像素，白色到淡蓝，带闪烁参数。
    """
    u = rng.uniform(-1.0, 1.0, n)
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    radius = rng.uniform(STAR_RADIUS[0], STAR_RADIUS[1], n)
    pos = np.stack([rxy * np.cos(th), u, rxy * np.sin(th)], axis=1) * radius[:, None]

    # 白色 → 淡蓝 之间平滑随机
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pale_blue = np.array([0.72, 0.85, 1.0], dtype=np.float32)
    t = rng.uniform(0.0, 1.0, n).astype(np.float32)[:, None]
    rgb = white * (1.0 - t) + pale_blue * t

    alpha = rng.uniform(STAR_ALPHA[0], STAR_ALPHA[1], n).astype(np.float32)
    col = np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32)

    size = np.rint(rng.uniform(STAR_SIZE_RANGE[0], STAR_SIZE_RANGE[1], n)
                   ).astype(np.float32)

    return {"pos": pos.astype(np.float32), "col": col, "size": size,
            "float_amp": 0.0,
            # 闪烁用的辅助数据
            "base_alpha": alpha.copy(),
            "twinkle_freq": rng.uniform(STAR_TWINKLE[0], STAR_TWINKLE[1],
                                        n).astype(np.float32),
            "twinkle_phase": rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)}


def make_nebula_particles(n=NEBULA_COUNT):
    """
    背景星云雾：用少量"大尺寸 + 极低透明度"的粒子铺一层很淡的紫色雾。
    它们远低于 Bloom 阈值，所以不会发光，只是把纯黑的背景垫起来一点。
    """
    # 分布在一个扁球壳里，铺满整个视野背景
    u = rng.uniform(-1.0, 1.0, n)
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    radius = rng.uniform(NEBULA_RADIUS[0], NEBULA_RADIUS[1], n)
    pos = np.stack([rxy * np.cos(th), u * 0.7, rxy * np.sin(th)], axis=1)
    pos = pos * radius[:, None]

    palette = np.asarray(NEBULA_COLORS, dtype=np.float32)
    pick = rng.integers(0, palette.shape[0], n)
    rgb = palette[pick] * rng.uniform(0.7, 1.0, n).astype(np.float32)[:, None]
    alpha = rng.uniform(NEBULA_ALPHA[0], NEBULA_ALPHA[1], n).astype(np.float32)

    size = random_sizes(n, NEBULA_SIZE_RANGE)
    # 显卡对 glPointSize 有上限，超了会被截断，这里先问一下能画多大
    try:
        hi = float(glGetFloatv(GL_POINT_SIZE_RANGE)[1])
        size = np.minimum(size, max(4.0, hi))
    except Exception:
        size = np.minimum(size, 32.0)

    return {"pos": pos.astype(np.float32),
            "col": np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32),
            "size": size, "float_amp": 0.0}


def make_confetti_particles(n=CONFETTI_COUNT):
    """
    彩色纸屑：对照 1.png，蛋糕周围撒着一层大小不一的圆彩点，
    左上角那一簇特别密。大小差别拉大，才有疏密错落的感觉。
    """
    n_spray = int(n * CONFETTI_SPRAY_RATIO)
    n_free = n - n_spray
    cx, cy, cz = CONFETTI_SPRAY

    # --- 左上角那一簇：球状聚集，越靠中心越密 ---
    u = rng.uniform(-1.0, 1.0, n_spray)
    th = rng.uniform(0.0, 2.0 * math.pi, n_spray)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    rad = CONFETTI_SPRAY_R * rng.uniform(0.0, 1.0, n_spray) ** 0.6   # 向中心聚
    spray = np.stack([rxy * np.cos(th), u, rxy * np.sin(th)], axis=1) * rad[:, None]
    spray += np.asarray((cx, cy, cz), dtype=np.float64)

    # --- 其余绕着蛋糕铺一圈 ---
    # 半径 1.3~3.4、上下 1.05 倍：绕着蛋糕的是一个**椭球壳**，
    # 前后左右上下的厚度差不多 —— 原先半径只给 1.7~2.6、上下压到 0.75，
    # 是一圈又薄又扁的带子，转到侧面时那一侧就几乎看不到彩点，
    # 于是"只有一块有粒子"的感觉又回来了（只不过换了一块）。
    u = rng.uniform(-1.0, 1.0, n_free)
    th = rng.uniform(0.0, 2.0 * math.pi, n_free)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    rad = rng.uniform(1.3, 3.4, n_free)
    free = np.stack([rxy * np.cos(th), u * 1.05, rxy * np.sin(th)], axis=1) * rad[:, None]

    pos = np.concatenate([spray, free], axis=0)

    palette = np.asarray(CONFETTI_COLORS, dtype=np.float32)
    pick = rng.integers(0, palette.shape[0], n)
    rgb = palette[pick] * rng.uniform(0.85, 1.0, n).astype(np.float32)[:, None]
    a = rng.uniform(CONFETTI_ALPHA[0], CONFETTI_ALPHA[1], n).astype(np.float32)

    return {"pos": pos.astype(np.float32),
            "col": np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32),
            "size": random_sizes(n, CONFETTI_SIZE_RANGE),
            "float_amp": FLOAT_AMP * 1.6}      # 纸屑飘得比蛋糕明显一点


def make_spark_particles(n=SPARK_COUNT):
    """
    四角星闪光：用星形贴图画出参考图里那些金色小星星。
    位置随机撒在蛋糕周围，尺寸偏大，靠贴图本身成形。
    """
    u = rng.uniform(-1.0, 1.0, n)
    th = rng.uniform(0.0, 2.0 * math.pi, n)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    # 半径收到 2.7 以内：相机距离下画面只有 ±2.9 宽、±2.3 高，
    # 撒到 3.6 的话有一半的星星落在画面外，白生成
    rad = rng.uniform(1.3, 2.7, n)
    pos = np.stack([rxy * np.cos(th), u * 0.8, rxy * np.sin(th)], axis=1) * rad[:, None]

    rgb = np.tile(np.asarray(SPARK_COLOR, dtype=np.float32), (n, 1))
    rgb *= rng.uniform(0.85, 1.0, n).astype(np.float32)[:, None]
    a = rng.uniform(SPARK_ALPHA[0], SPARK_ALPHA[1], n).astype(np.float32)

    return {"pos": pos.astype(np.float32),
            "col": np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32),
            "size": random_sizes(n, SPARK_SIZE_RANGE),
            "float_amp": FLOAT_AMP * 2.0,
            # 星形闪光要一闪一闪
            "base_alpha": a.copy(),
            "twinkle_freq": rng.uniform(0.8, 2.0, n).astype(np.float32),
            "twinkle_phase": rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)}


def make_flame_particles(n=FLAME_COUNT):
    """
    火焰：上尖下宽的水滴形，黄→橙→红渐变。
    用 life(0→1) 循环实现"向上飘散 → 顶部消失 → 底部重生"。
    """
    life = rng.uniform(0.0, 1.0, n).astype(np.float32)
    th = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
    rad = np.sqrt(rng.uniform(0.0, 1.0, n)).astype(np.float32)
    rate = rng.uniform(FLAME_RISE[0], FLAME_RISE[1], n).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)

    # 预先算好每颗粒子按 life 取色的结果（底部黄 → 中部橙 → 顶部红）
    bot = np.asarray(FLAME_COL_BOTTOM, dtype=np.float32)
    mid = np.asarray(FLAME_COL_MID, dtype=np.float32)
    top = np.asarray(FLAME_COL_TOP, dtype=np.float32)
    t = life[:, None]
    base_rgb = np.where(t < 0.5,
                        bot + (mid - bot) * (t / 0.5),
                        mid + (top - mid) * ((t - 0.5) / 0.5)).astype(np.float32)

    return {
        "n": n,
        "life": life, "theta": th, "rad": rad, "rate": rate, "phase": phase,
        "base_rgb": base_rgb,
        "pos": np.zeros((n, 3), dtype=np.float32),
        "col": np.zeros((n, 4), dtype=np.float32),
        "size": random_sizes(n, FLAME_SIZE_RANGE),
        "float_amp": 0.0,
    }


def update_flame(flame, t, dt):
    """每帧更新火焰的位置和颜色：上浮、抖动、闪烁、到顶后回到底部重生"""
    life = flame["life"]
    life += flame["rate"] * dt
    np.mod(life, 1.0, out=life)                 # 到顶就回到底部重新生成

    h = life                                    # 0 = 底部, 1 = 顶部
    th = flame["theta"]
    ph = flame["phase"]

    fx, fy, fz = FLAME_POS

    # 水滴形：贴着烛芯的底部先收一下，往上鼓起来，再一路收窄聚成尖
    bulge = 0.55 + 0.45 * np.minimum(1.0, h / 0.15)
    r = (FLAME_BASE_R * flame["rad"]
         * np.power(1.0 - h, FLAME_TAPER) * bulge
         + FLAME_DRIFT * h ** 2.0)

    # 绕轴旋转 + 随机抖动
    ang = th + t * FLAME_SWIRL + h * 2.0
    flame["pos"][:, 0] = fx + np.cos(ang) * r + FLAME_JITTER * np.sin(t * 17.0 + ph)
    flame["pos"][:, 1] = fy + h * FLAME_HEIGHT
    flame["pos"][:, 2] = fz + np.sin(ang) * r + FLAME_JITTER * np.cos(t * 19.0 + ph)

    # 透明度：底部快速点亮，越往上越淡直至消失
    fade_in = np.clip(life / 0.10, 0.0, 1.0)
    fade_out = 1.0 - np.clip((life - 0.45) / 0.55, 0.0, 1.0) ** 1.6
    a = fade_in * fade_out * rng.uniform(FLAME_ALPHA[0], FLAME_ALPHA[1], flame["n"])

    flame["col"][:, :3] = flame["base_rgb"]
    flame["col"][:, 3] = np.clip(a, 0.0, 1.0)


# ==============================================================================
# ④ VBO 创建 / 粒子系统
# ==============================================================================

def make_glow_texture(size=GLOW_TEX_SIZE):
    """
    程序生成一张"柔光球"贴图：中间实心、边缘渐隐，没有任何外部图片。
    配合 GL_POINT_SPRITE + GL_COORD_REPLACE 贴到每个点上，
    方形的点就变成了柔和的小光球（这正是"边缘有透明度渐变"的来源）。
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) * 0.5
    r = np.clip(np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c, 0.0, 1.0)

    # 渐变必须铺满整个贴图半径：点只有几像素大，如果只在最外圈留一点点渐变，
    # 缩小采样之后渐变会被整个跳过，粒子看起来还是硬边方块。
    halo = np.clip(1.0 - r * r, 0.0, 1.0) ** GLOW_HALO               # 全半径平滑衰减
    core = np.clip((GLOW_CORE - r) / GLOW_CORE, 0.0, 1.0) ** 0.85    # 中心实心核
    a = np.clip(halo * 0.80 + core * 0.45, 0.0, 1.0)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = 255                       # 白色，靠 GL_MODULATE 染成粒子颜色
    rgba[..., 3] = (a * 255.0).astype(np.uint8)
    data = np.ascontiguousarray(rgba)

    tex = _gl_id(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    # 贴图要缩到几个像素去画，必须用 mipmap 做缩小过滤，
    # 否则 GL_LINEAR 只采样足迹中心的一小块，渐变等于不存在。
    try:
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR)
    except Exception:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def make_star_texture(size=STAR_TEX_SIZE):
    """
    程序生成一张四角星贴图（参考图里散落的金色小星星）。
    做法：一个实心核 + 横向/纵向两条细芒，取最大值合成。
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) * 0.5
    dx = (xx - c) / c                      # -1 ~ 1
    dy = (yy - c) / c
    r = np.sqrt(dx * dx + dy * dy)

    core = np.clip(1.0 - r / STAR_TEX_CORE, 0.0, 1.0)
    # 芒：横向那条要求 |dy| 很窄、|dx| 沿半径衰减
    arm_h = (np.clip(1.0 - np.abs(dy) / STAR_TEX_ARM, 0.0, 1.0) ** 0.6
             * np.clip(1.0 - np.abs(dx), 0.0, 1.0) ** STAR_TEX_FALLOFF)
    arm_v = (np.clip(1.0 - np.abs(dx) / STAR_TEX_ARM, 0.0, 1.0) ** 0.6
             * np.clip(1.0 - np.abs(dy), 0.0, 1.0) ** STAR_TEX_FALLOFF)
    # 横截面的 0.6 次方把线性衰减"撑方"：星芒中间一段接近满强度，
    # 到边上才陡然掉下去。纯线性的话星芒的**平均**亮度只有峰值一半，
    # 再经一次缩小采样就基本看不见了。
    a = np.clip(np.maximum(np.maximum(core, arm_h), arm_v), 0.0, 1.0)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (a * 255.0).astype(np.uint8)
    data = np.ascontiguousarray(rgba)

    tex = _gl_id(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    # 这张贴图**故意不做 mipmap**。柔光球（GLOW_TEX）必须做 —— 它是一团
    # 没有细节的晕，mipmap 让缩小时更平滑；但星芒是细特征，
    # mipmap 的本质就是"缩小时把邻域平均掉"，平均掉的正是那两条芒。
    # 点上屏之后只有 9~22 像素，本来就在缩小采样，再走 mipmap 就只剩核了。
    # 不生成 mipmap 时必须显式设 MIN_FILTER：默认值是 GL_NEAREST_MIPMAP_LINEAR，
    # 那会去采样一层根本不存在的 mipmap，整张贴图直接失效。
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


# 点精灵贴图（在 init_point_sprites() 里创建）
GLOW_TEX = None
STAR_TEX = None
POINT_SPRITE_OK = False


def init_point_sprites():
    """
    开启固定管线的点精灵：让每个 GL_POINTS 用 glPointCoord 采样贴图，
    从而把方形点渲染成柔光球 / 四角星。驱动不支持时退回普通圆点。
    """
    global GLOW_TEX, STAR_TEX, POINT_SPRITE_OK

    if not GLOW_TEX:
        GLOW_TEX = make_glow_texture()
    if not STAR_TEX:
        STAR_TEX = make_star_texture()
    try:
        glEnable(GL_POINT_SPRITE)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, GLOW_TEX)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        POINT_SPRITE_OK = True
    except Exception:
        POINT_SPRITE_OK = False
    print("点精灵柔光贴图:", "开启" if POINT_SPRITE_OK else "不可用，退回圆点")
    return POINT_SPRITE_OK


class ParticleSystem:
    """
    一组粒子。位置 / 颜色各一个 VBO，用
    glVertexPointer + glColorPointer + glDrawArrays(GL_POINTS) 绘制。
    粒子大小不同时按大小分档，每档一次 glPointSize + glDrawArrays。
    """

    def __init__(self, data, use_star_tex=False,
                 solid=False, depth_write=True, depth_test=True):
        pos = np.ascontiguousarray(data["pos"], np.float32)
        col = np.ascontiguousarray(data["col"], np.float32)
        size = np.asarray(data["size"], np.float32)
        if size.ndim == 0:
            size = np.full(pos.shape[0], float(size), np.float32)

        self.n = int(pos.shape[0])
        self.float_amp = float(data.get("float_amp", 0.0))
        # 需要每帧重算几何的系统（目前只有光环）把自己的参数放在这里，
        # 由外面那个专门的 anim 函数读写。放这儿是为了让它跟粒子数据
        # 一起被构造出来，不用在外面另挂一份状态。
        self.anim = data.get("anim")
        # （原先这里有个 camera_relative 开关，注释写着"星空不跟随蛋糕自转"。
        #   但它只在构造函数里存了一下，draw() 从头到尾没读过 —— 一个不起作用的
        #   开关，却让人以为"背景不跟着转"是有意为之，反而把真正的原因
        #   （几个氛围系统画在了 glPushMatrix 外面）藏起来了。删掉。）
        self.use_star_tex = use_star_tex           # 四角星闪光用星形贴图
        # solid：蛋糕 / 盘子 / 光环 / 蜡烛这类"实体面"，用不贴图的方点，
        # 每颗粒子铺满自己的方块。原因见文件上方"实体面粒子"那段说明。
        self.solid = solid
        # 深度策略：
        #   depth_test   关掉 = 永远画在最前面（火焰要这种，否则会被自己掏空成一层壳）
        #   depth_write  关掉 = 会被别的东西挡住，但自己不挡别人（星云雾这类薄雾要这种）
        self.depth_write = depth_write
        self.depth_test = depth_test

        # ---------- 边缘加强（只在渲染阶段做，不改粒子生成的结果）----------
        # 越靠蛋糕外侧的粒子越大越亮，中心稍暗，形成"星云包裹"的观感。
        if data.get("edge_emphasis"):
            rad = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
            top = float(rad.max())
            if top > 1e-6:
                rn = rad / top                      # 0 = 中轴, 1 = 最外圈
                size = size * (1.0 + EDGE_SIZE_BOOST * rn)
                # 只提亮不压暗，避免中心粒子被削到看不见
                col[:, 3] = np.clip(col[:, 3] * (1.0 + EDGE_ALPHA_BOOST * rn), 0.0, 1.0)

        # ---------- 实体面：把 alpha 预先乘进颜色 ----------
        # 实体用覆盖式混合画（GL_ONE / GL_ZERO），那个模式下混合方程只认
        # 颜色通道、不看 alpha，所以亮度必须提前乘进去，否则所有粒子都是满亮度。
        if self.solid:
            col[:, :3] *= col[:, 3:4]

        # ---------- 按粒子大小分档 ----------
        order = np.argsort(size, kind="stable")
        pos, col, size = pos[order], col[order], size[order]
        self.groups = []                            # (起始下标, 数量, 像素大小)
        start = 0
        for i in range(1, self.n + 1):
            if i == self.n or size[i] != size[start]:
                self.groups.append((start, i - start, float(size[start])))
                start = i

        self.base_pos = pos.copy()                  # 原始位置（浮动动画的基准）
        self.pos = pos
        self.col = col

        # ---------- 每颗粒子的浮动参数（频率随机）----------
        self.phase = rng.uniform(0.0, 2.0 * math.pi, self.n).astype(np.float32)
        self.freq = rng.uniform(FLOAT_FREQ[0], FLOAT_FREQ[1], self.n).astype(np.float32)

        # ---------- 创建 VBO ----------
        self.buf_pos = self._make_buffer(pos)
        self.buf_col = self._make_buffer(col)

    @staticmethod
    def _make_buffer(arr):
        buf = _gl_id(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, buf)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return buf

    def _upload(self, buf, arr):
        glBindBuffer(GL_ARRAY_BUFFER, buf)
        glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ------------------------------ 每帧动画 ------------------------------
    def update_float(self, t):
        """轻微的上下浮动，幅度 FLOAT_AMP，频率每颗粒子随机"""
        if self.float_amp <= 0.0:
            return
        self.pos[:, 1] = self.base_pos[:, 1] + \
            np.sin(t * self.freq + self.phase) * self.float_amp
        self._upload(self.buf_pos, self.pos)

    def update_positions(self, pos):
        self.pos = np.ascontiguousarray(pos, np.float32)
        self._upload(self.buf_pos, self.pos)

    def update_colors(self, col):
        self.col = np.ascontiguousarray(col, np.float32)
        self._upload(self.buf_col, self.col)

    def update_twinkle(self, t, base_alpha, freq, phase):
        """星星闪烁：只改 alpha 通道"""
        self.col[:, 3] = base_alpha * (0.55 + 0.45 * np.sin(t * freq + phase))
        self._upload(self.buf_col, self.col)

    # ------------------------------ 绘制 ------------------------------
    def draw(self):
        if self.solid:
            # 实体面：不贴图、不做点平滑 —— 就是纯方的实心点，方块填满方块。
            # GL_POINT_SMOOTH 也必须关：它会把方点削成圆点，
            # 四角变成透明的，洞又回来了。
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_POINT_SMOOTH)
        else:
            # 发光的小点：绑柔光球 / 星形贴图，边缘是渐隐的。
            tex = STAR_TEX if self.use_star_tex else GLOW_TEX
            glEnable(GL_POINT_SMOOTH)
            if POINT_SPRITE_OK and tex:
                glActiveTexture(GL_TEXTURE0)
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex)
                glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
            else:
                glDisable(GL_TEXTURE_2D)

        if self.depth_test:
            glEnable(GL_DEPTH_TEST)
        else:
            glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE if self.depth_write else GL_FALSE)

        # 实体：覆盖式写（dst = src）。配合 GL_LESS + 深度写，
        # 每个像素最终就是那颗最近的粒子，和绘制顺序无关。
        # 发光的小点：照旧加法混合，重叠越多越亮。
        if self.solid:
            glBlendFunc(GL_ONE, GL_ZERO)
        else:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.buf_pos)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.buf_col)
        glColorPointer(4, GL_FLOAT, 0, None)

        for start, count, px in self.groups:
            glPointSize(px)
            glDrawArrays(GL_POINTS, start, count)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


# ==============================================================================
# ⑤ Bloom 后处理（FBO + 可分离高斯模糊，纯固定管线实现）
# ==============================================================================

class Framebuffer:
    """离屏渲染目标：一张纹理 + 一个 FBO（可选带深度缓冲）"""

    def __init__(self, w, h, with_depth=False):
        self.w, self.h = int(w), int(h)
        self.depth = None
        self.tex = _gl_id(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.w, self.h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)

        self.fbo = _gl_id(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.tex, 0)

        # 场景 FBO 必须挂一个深度附件。只挂颜色纹理的话，glEnable(GL_DEPTH_TEST)
        # 是**空操作** —— 没有深度缓冲可测，遮挡完全不生效，蛋糕背面的粒子
        # 照样叠在正面顶面上，光环也会从蛋糕里透出来。
        if with_depth:
            self.depth = _gl_id(glGenRenderbuffers(1))
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                                  self.w, self.h)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, self.depth)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("帧缓冲创建失败，显卡不支持 FBO")

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.w, self.h)


class BloomPipeline:
    """
    辉光管线（全部固定管线，不使用着色器）：

        粒子 → scene(FBO)
                 ├─(亮度提取)→ bright ─(横模糊)→ blur_a ─(纵模糊)→ blur_b   ← 近层
                 └─(降采样)──→ far_a  ─(横模糊)→ far_b  ─(纵模糊)→ far_a    ← 远层(半分辨率)

        屏幕 = scene×0.5 + blur_b×0.3 + far_a×0.3

    两个要点：
      · 原图必须单独留一份。模糊如果在同一个 FBO 里"就地乒乓"回去，
        原图就被覆盖了，最后没有清晰的底图可叠（这是常见写法的一个坑）。
      · 远层用半分辨率做模糊：同样的 sigma 等效半径翻倍，而且便宜 4 倍。
        宽光晕本来就没有高频细节，降采样看不出来。
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.scene = Framebuffer(w, h, with_depth=True)
        self.bright = Framebuffer(w, h)
        self.blur1_a = Framebuffer(w, h)          # 近层，全分辨率
        self.blur1_b = Framebuffer(w, h)
        hw, hh = max(1, w // 2), max(1, h // 2)
        self.far_a = Framebuffer(hw, hh)          # 远层，半分辨率
        self.far_b = Framebuffer(hw, hh)

        self.wn, self.halfn = gaussian_weights(BLOOM_SIGMA_NEAR)
        self.wf, self.halff = gaussian_weights(BLOOM_SIGMA_FAR)

        # 全屏四边形 VBO（两个三角形），位置和纹理坐标各一个 buffer
        quad = np.array([-1, -1, 1, -1, 1, 1,
                         -1, -1, 1, 1, -1, 1], dtype=np.float32)
        uv = np.array([0, 0, 1, 0, 1, 1,
                       0, 0, 1, 1, 0, 1], dtype=np.float32)
        self.quad_pos = ParticleSystem._make_buffer(quad)
        self.quad_uv = ParticleSystem._make_buffer(uv)

    # ------------------------------ 基础绘制 ------------------------------
    def draw_quad(self):
        """
        画一个覆盖整个 FBO 的四边形（走 VBO，不用立即模式）。
        注意：这里必须把投影/模型视图矩阵临时复位成单位矩阵 —— 后处理阶段
        主场景的透视矩阵还挂在矩阵栈上，若沿用它，-1~1 的四边形盖不满屏幕。
        """
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_pos)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_uv)
        glTexCoordPointer(2, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _blit(self, target, texture, color=(1.0, 1.0, 1.0, 1.0)):
        """把一个纹理原样画到目标 FBO 上（替换内容）"""
        target.bind()
        self._draw_tex(texture, color, blend=False)

    def _draw_tex(self, texture, color, blend):
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glColor4f(*color)

        if blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)
        else:
            glDisable(GL_BLEND)

        self.draw_quad()
        glDisable(GL_TEXTURE_2D)

    # ------------------------------ 各阶段 ------------------------------
    def _blur_pass(self, target, source, dx, dy, weights, half):
        """
        一遍可分离高斯模糊。
        做法：把源纹理**画 N 次**，每次用纹理矩阵偏移一个像素、
        颜色乘以对应高斯权重，用加法混合累加 —— 等价于一次卷积。
        """
        target.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)                 # 纯加法累加
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, source)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # 偏移量按**目标 FBO 自己的分辨率**算（远层是半分辨率）
        texel_x = 1.0 / target.w
        texel_y = 1.0 / target.h

        for i, w in enumerate(weights):
            off = (i - half) * 1.0                  # 偏移的像素数
            # 注意：每次都要重新切到纹理矩阵 —— draw_quad() 结束时会把当前
            # 矩阵模式留回 GL_MODELVIEW，只在循环外设一次的话，从第二轮起
            # 偏移就全打到模型视图矩阵上了，卷积会退化成"同一张图叠 N 次"。
            glMatrixMode(GL_TEXTURE)
            glLoadIdentity()
            glTranslatef(off * dx * texel_x, off * dy * texel_y, 0.0)
            glColor4f(float(w), float(w), float(w), 1.0)
            self.draw_quad()
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_TEXTURE_2D)

    def run(self):
        """
        执行完整的辉光管线，最后合成到屏幕（FBO 0）：

            bright  = clamp(场景 × 增益 − 阈值, 0, 1)     ← 只留最亮的粒子
            近层模糊 = 高斯(bright, sigma 3)              横向一遍 + 纵向一遍
            远层模糊 = 高斯(bright, sigma 9)
            屏幕    = 原图×0.5 + 近层×0.3 + 远层×0.3

        注意原图必须单独留一份：模糊是"就地乒乓"回去的话会把原图覆盖掉，
        最后就没有清晰的底图可以叠加了（FBO_A →FBO_B →FBO_A 的写法有这个坑）。
        """
        # 后处理是纯 2D 贴图合成，全程不需要深度测试；不关掉的话，
        # 合成用的方片和文字会被场景深度缓冲挡住。
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        if not BLOOM_ENABLED:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, self.w, self.h)
            self._draw_tex(self.scene.tex, (1.0, 1.0, 1.0, 1.0), blend=False)
            return

        # --- 1. 亮度提取：先乘增益，再减阈值，结果 clamp 到 0 ---
        g = BLOOM_GAIN
        self._blit(self.bright, self.scene.tex, (g, g, g, 1.0))
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)              # dst = dst - src
        glDisable(GL_TEXTURE_2D)                               # 画纯色做减法
        glColor4f(BLOOM_THRESHOLD, BLOOM_THRESHOLD, BLOOM_THRESHOLD, 1.0)
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        self.draw_quad()
        glBlendEquation(GL_FUNC_ADD)

        # --- 2. 近层：全分辨率，先横向再纵向 ---
        self._blur_pass(self.blur1_a, self.bright.tex, 1.0, 0.0, self.wn, self.halfn)
        self._blur_pass(self.blur1_b, self.blur1_a.tex, 0.0, 1.0, self.wn, self.halfn)

        # --- 3. 远层：降到半分辨率再模糊（等效半径翻倍，开销降到 1/4）---
        self._blit(self.far_a, self.bright.tex)          # 带 LINEAR 缩小的降采样
        self._blur_pass(self.far_b, self.far_a.tex, 1.0, 0.0, self.wf, self.halff)
        self._blur_pass(self.far_a, self.far_b.tex, 0.0, 1.0, self.wf, self.halff)

        # --- 4. 加权合成到屏幕 ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.w, self.h)

        a0 = BLOOM_W_ORIGINAL
        self._draw_tex(self.scene.tex, (a0, a0, a0, 1.0), blend=False)

        a1 = BLOOM_W_NEAR
        self._draw_tex(self.blur1_b.tex, (a1, a1, a1, 1.0), blend=True)

        a2 = BLOOM_W_FAR
        self._draw_tex(self.far_a.tex, (a2, a2, a2, 1.0), blend=True)


# ==============================================================================
# ⑥ 文字（Pygame 渲染 → 贴图 → 正交投影贴屏）
# ==============================================================================

def build_text_surface(text, size, color=TEXT_COLOR, glow_rgb=TEXT_GLOW_RGB):
    font = pygame.font.SysFont(FONT_NAME, size, bold=True)
    core = font.render(text, True, color)
    w, h = core.get_size()
    pad = max(8, size // 3)

    surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    glow = font.render(text, True, tuple(glow_rgb))
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        surf.blit(glow, (pad + dx, pad + dy), special_flags=pygame.BLEND_RGBA_ADD)

    small = pygame.transform.smoothscale(
        pygame.transform.smoothscale(
            surf, (max(1, surf.get_width() // 8), max(1, surf.get_height() // 8))),
        surf.get_size())
    surf.blit(small, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    surf.blit(core, (pad, pad))
    return surf


def surface_to_texture(surf):
    w, h = surf.get_size()
    to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    data = to_bytes(surf, "RGBA", True)

    tex = _gl_id(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex, w, h


def draw_text_overlay(tex, w, h, x, y):
    """在屏幕上贴一张 2D 贴图（VBO + 正交投影）"""
    glMatrixMode(GL_TEXTURE)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    quad = np.array([x, y, x + w, y, x + w, y + h,
                     x, y, x + w, y + h, x, y + h], dtype=np.float32)
    uv = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=np.float32)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexPointer(2, GL_FLOAT, 0, quad)
    glTexCoordPointer(2, GL_FLOAT, 0, uv)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisable(GL_TEXTURE_2D)


# ==============================================================================
# ⑦ 渲染 / 主循环
# ==============================================================================

def save_screenshot(path):
    glReadBuffer(GL_BACK)
    data = glReadPixels(0, 0, WIN_W, WIN_H, GL_RGBA, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(WIN_H, WIN_W, 4)[::-1]
    surf = pygame.image.frombuffer(np.ascontiguousarray(img), (WIN_W, WIN_H), "RGBA")
    pygame.image.save(surf, path)
    print("已保存截图:", path)


def main(screenshot=False):
    # ------------------------------ 初始化 ------------------------------
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    print("OpenGL :", glGetString(GL_VERSION).decode(errors="replace"))
    print("GPU    :", glGetString(GL_RENDERER).decode(errors="replace"))

    # ------------------------------ OpenGL 状态 ------------------------------
    # 深度测试要开。加法混合不排序也能画，但**遮挡**是另一回事：
    # 蛋糕背面的粒子会和正面的顶面叠在同一个像素上，把金色顶面冲成灰白；
    # 光环落在蛋糕后面的那一半也会透过来。开了深度测试，这两件事才成立。
    #
    # 函数必须是 GL_LESS，不能是 GL_LEQUAL。
    # 蛋糕侧面是一层薄壳，一个像素的足迹范围内几十颗粒子的深度**几乎相等**；
    # 用 LEQUAL 的话"相等也算通过"，这几十颗会全部叠加上去，整个蛋糕直接过曝，
    # 环也糊成一条白色宽带。LESS 严格小于：等深的后到者被挡掉，
    # 每个像素最终只留下最前面那一颗 —— 表面亮度就由粒子颜色说了算，
    # 也就是由大理石纹决定，而不是由"这一像素上碰巧叠了几颗"决定。
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glDepthMask(GL_TRUE)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)           # 加法混合，重叠粒子自然变亮
    glEnable(GL_POINT_SMOOTH)                   # 圆点粒子
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glShadeModel(GL_SMOOTH)
    glClearColor(*BG_COLOR)

    init_point_sprites()                        # 柔光小光球贴图

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = WIN_W / float(WIN_H)
    f = 1.0 / math.tan(math.radians(FOV) * 0.5)
    glFrustum(-NEAR * aspect / f, NEAR * aspect / f,
              -NEAR / f, NEAR / f, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)

    # ------------------------------ 生成粒子 ------------------------------
    cake_data = make_cake_particles()
    plate_data = make_plate_particles()
    ring_data = make_ring_particles()
    star_data = make_star_particles()
    confetti_data = make_confetti_particles()
    spark_data = make_spark_particles()
    flame = make_flame_particles()
    candle_data = make_candle_particles()
    nebula_data = make_nebula_particles()

    # ------------------------------ 创建 VBO ------------------------------
    # 实体组：写深度，互相遮挡（蛋糕挡住背面的粒子、挡住盘子的后半圈、挡住环的后半圈）。
    # 一律用平顶圆盘贴图 —— 这几组是"面"，不是"发光的点"。
    sys_cake = ParticleSystem(cake_data, solid=True)
    sys_plate = ParticleSystem(plate_data, solid=True)
    # 环是**发光**的，不是实体：柔光球贴图 + 加法混合。
    # 它照旧接受深度测试（蛋糕后面那半圈会被判掉），但不写深度 ——
    # 它是光，不该把后面的东西挡掉。
    sys_ring = ParticleSystem(ring_data, depth_write=False)
    sys_candle = ParticleSystem(candle_data, solid=True)
    # 氛围组：接受遮挡，但不遮挡别人（薄雾/星点不该挡掉蛋糕）。
    # 它们和蛋糕一样在 glRotatef(angle) 里画，所以跟着一起自转。
    sys_stars = ParticleSystem(star_data, depth_write=False)
    sys_confetti = ParticleSystem(confetti_data, depth_write=False)
    sys_spark = ParticleSystem(spark_data, use_star_tex=True, depth_write=False)
    sys_nebula = ParticleSystem(nebula_data, depth_write=False)
    # 火焰：关掉深度测试。它是一团体积，写深度的话会被自己掏成一层空壳。
    sys_flame = ParticleSystem({
        "pos": flame["pos"], "col": flame["col"],
        "size": flame["size"], "float_amp": 0.0,
    }, depth_write=False, depth_test=False)

    total = (sys_cake.n + sys_plate.n + sys_ring.n + sys_stars.n
             + sys_confetti.n + sys_spark.n
             + sys_flame.n + sys_candle.n + sys_nebula.n)
    print(f"粒子总数: {total}  "
          f"(蛋糕 {sys_cake.n} / 底盘 {sys_plate.n} / 光环 {sys_ring.n} "
          f"/ 星星 {sys_stars.n} / 纸屑 {sys_confetti.n} / 星闪 {sys_spark.n} "
          f"/ 火焰 {sys_flame.n} / 蜡烛 {sys_candle.n} / 星云 {sys_nebula.n})")

    # ------------------------------ Bloom 后处理 ------------------------------
    bloom = BloomPipeline(WIN_W, WIN_H)

    # ------------------------------ 文字 ------------------------------
    tex_main, tw_main, th_main = surface_to_texture(
        build_text_surface(TEXT_MAIN, TEXT_MAIN_SIZE))

    # ------------------------------ 状态 ------------------------------
    angle = 0.0
    ring_angle = 0.0
    cam_yaw = 0.0
    cam_pitch, cam_dist = CAM_PITCH_START, CAM_DIST_START
    dragging, last_mouse = False, (0, 0)
    t0 = pygame.time.get_ticks() * 0.001
    frames = 0

    star_base_alpha = star_data["base_alpha"]
    star_freq = star_data["twinkle_freq"]
    star_phase = star_data["twinkle_phase"]
    spark_base_alpha = spark_data["base_alpha"]
    spark_freq = spark_data["twinkle_freq"]
    spark_phase = spark_data["twinkle_phase"]

    try:
        # ============================== 主循环 ==============================
        while True:
            dt = clock.tick(FPS) * 0.001
            t = pygame.time.get_ticks() * 0.001 - t0

            # ------------------------------ 事件 ------------------------------
            for event in pygame.event.get():
                if event.type == QUIT:
                    raise SystemExit
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    raise SystemExit
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging, last_mouse = True, event.pos
                    elif event.button == 4:
                        cam_dist = max(CAM_DIST_MIN, cam_dist - ZOOM_SENS)
                    elif event.button == 5:
                        cam_dist = min(CAM_DIST_MAX, cam_dist + ZOOM_SENS)
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                elif event.type == MOUSEMOTION and dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    last_mouse = event.pos
                    cam_yaw += dx * DRAG_SENS
                    cam_pitch = min(CAM_PITCH_MAX,
                                    max(CAM_PITCH_MIN, cam_pitch + dy * DRAG_SENS))
                elif event.type == MOUSEWHEEL:
                    cam_dist = min(CAM_DIST_MAX,
                                   max(CAM_DIST_MIN, cam_dist - event.y * ZOOM_SENS))

            # ------------------------------ 动画更新 ------------------------------
            angle = (angle + ROT_SPEED * dt) % 360.0
            ring_angle = (ring_angle + RING_SPIN * dt) % 360.0

            sys_cake.update_float(t)                      # 上下浮动
            ring_animate(sys_ring, t)                     # 光环：行波，不是随机抖
            sys_confetti.update_float(t)
            sys_spark.update_float(t)
            sys_stars.update_twinkle(t, star_base_alpha, star_freq, star_phase)
            sys_spark.update_twinkle(t, spark_base_alpha, spark_freq, spark_phase)

            update_flame(flame, t, dt)                    # 火焰上浮 + 抖动
            sys_flame.update_positions(flame["pos"])
            sys_flame.update_colors(flame["col"])

            # ------------------------- 渲染到场景 FBO -------------------------
            bloom.scene.bind()
            glClearColor(*BG_COLOR)
            # 清深度之前必须先把深度写打开。上一帧最后画的是火焰（depth_write=False），
            # 它把 glDepthMask 留在 GL_FALSE 上；此时 glClear 是清不掉深度缓冲的，
            # 深度值会一帧帧累积，最后整个蛋糕都被自己的历史深度挡掉、直接消失。
            glDepthMask(GL_TRUE)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -cam_dist)
            glRotatef(cam_pitch, 1.0, 0.0, 0.0)
            glRotatef(cam_yaw, 0.0, 1.0, 0.0)

            # 先画写深度的实体，深度缓冲里有了它们，后面那些"不写深度"的
            # 氛围粒子才能被正确挡住（比如蛋糕背后的星星和纸屑）。
            # 蛋糕 + 底盘 + 蜡烛 + 光环：一起绕 Y 轴自转
            glPushMatrix()
            glRotatef(angle, 0.0, 1.0, 0.0)

            # 顺序有讲究：蛋糕先画并写深度，盘子再画。
            # 盘子的**后半圈**落在蛋糕后面，深度测试会把它判掉；
            # 如果反过来先画盘子，那半圈会先加进颜色缓冲，
            # 加法混合不遮挡，它就会透过蛋糕叠上来，把蛋糕下段冲白。
            sys_cake.draw()
            sys_plate.draw()
            sys_candle.draw()

            # 氛围粒子：星云雾 + 星星 + 纸屑 + 星闪。
            # **必须画在这个 glPushMatrix 里面**，也就是跟着蛋糕一起绕 Y 轴转。
            # 原先它们画在 glPopMatrix 之后：蛋糕在自转，背景的星点、星云、
            # 左上角那一簇纸屑却纹丝不动 —— 看起来就是"蛋糕在一个静止的幕布前面转"，
            # 幕布感一出来，整幅画的空间关系就塌了。
            # 位置仍然在蛋糕之后画，所以深度测试照旧有效（蛋糕背后的粒子会被挡住）。
            sys_nebula.draw()
            sys_stars.draw()
            sys_confetti.draw()
            sys_spark.draw()

            glPopMatrix()

            # 光环：**反着转**（RING_SPIN 是负的）。
            # 单独一个 glPushMatrix 是因为它要的旋转角跟蛋糕不一样 ——
            # 倾角本身已经烘进粒子坐标了（每条轨道一个角度，一个 glRotatef 给不了），
            # 能在这里统一的只有"绕 Y 轴转多少"这一个角度。
            # 拆出来不破坏遮挡：环不写深度，但照旧接受深度测试，
            # 蛋糕后面那半圈还是会被深度缓冲判掉。
            glPushMatrix()
            glRotatef(ring_angle, 0.0, 1.0, 0.0)
            sys_ring.draw()
            glPopMatrix()

            # 火焰：不参与深度，永远压在最上面
            glPushMatrix()
            glRotatef(angle, 0.0, 1.0, 0.0)
            sys_flame.draw()
            glPopMatrix()

            # ------------------------- Bloom 后处理 + 合成 -------------------------
            bloom.run()

            # ------------------------------ 文字 ------------------------------
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, WIN_W, 0, WIN_H, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            draw_text_overlay(tex_main, tw_main, th_main,
                              (WIN_W - tw_main) * 0.5, TEXT_MARGIN_BOTTOM)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            pygame.display.flip()

            # ------------------------------ 调试截图 ------------------------------
            frames += 1
            if screenshot and frames >= 90:
                save_screenshot("preview.png")
                raise SystemExit

    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main(screenshot="--shot" in sys.argv)
    except SystemExit:
        pass
    except Exception as exc:
        print("\n运行出错:", exc)
        print("依赖安装:  python -m pip install pygame-ce PyOpenGL numpy")
        sys.exit(1)
