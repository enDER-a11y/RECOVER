import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 修正后数据（872份样本）
labels = ['A.男', 'B.女']
sizes = [48.7, 51.3]  # 修正为目标占比
colors = ['#6699ff', '#c6d8ff']  # 深蓝/浅蓝，与原图配色一致

# 绘制环形饼图
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.2f%%',  # 保留2位小数，可改为%1.1f%%显示1位
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='white')  # 环形样式
)

# 调整标注字体
plt.setp(texts, fontsize=16)
plt.setp(autotexts, fontsize=14, weight='bold')

# 添加图例
plt.legend(wedges, labels, loc="lower center", ncol=2, fontsize=16, frameon=False)

# 保证饼图为正圆
plt.axis('equal')
plt.title('受访者性别分布', fontsize=18, pad=20)

plt.show()