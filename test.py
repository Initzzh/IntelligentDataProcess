import matplotlib.pyplot as plt

# 创建画布和主图
fig = plt.figure(figsize=(12, 8))
# ax_main = fig.add_subplot(1, 1, 1)

# 创建子图1，并指定位置和大小
ax_sub1 = fig.add_axes([0.12,0.72,0.28, 0.1])

# 创建子图2，并指定位置和大小
ax_sub2 = fig.add_axes([0.12 ,0.54, 0.28, 0.1])
ax_sub2.set_title("ax_sub2")

# 创建子图3，并指定位置和大小
ax_sub3 = fig.add_axes([0.12,0.36, 0.28,0.1])

# 创建子图4，并指定位置和大小
ax_sub4 = fig.add_axes([0.12, 0.18, 0.28, 0.12])

# 创建子图5，并指定位置和大小
ax_sub5 = fig.add_axes([0.58, 0.54, 0.28 , 0.3])

# 添加文字
ax_sub7 = fig.add_axes([0.85,0.54,0.1,0.2])
ax_sub7.text( 0.8, 0.54, s='Line 1\nLine 2\nLine 3', fontsize=20, ha='center', va='center', multialignment='center')
ax_sub7.axis('off')
# 添加表格
ax_sub6 =  fig.add_axes([0.58,0.18,0.4,0.3])

cell_text = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
table = ax_sub6.table(cellText=cell_text, loc='center')
# 隐藏坐标轴
ax_sub6.axis('off')

# 在主图和子图中绘制曲线
x = [1, 2, 3, 4, 5]
y1 = [1, 2, 3, 4, 5]
y2 = [5, 4, 3, 2, 1]
ax_sub1.plot(x, y1)
ax_sub2.plot(x, y1)
ax_sub3.plot(x, y1)
ax_sub4.plot(x, y1)
ax_sub5.plot(x, y1)
# ax_main.plot(x, y2)

# 显示图形
plt.show()

T= 27
text = "环境温度:"+str(T)
print(text)