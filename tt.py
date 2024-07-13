import numpy as np

# 示例数组
arr = np.array([
    [4, 2],
    [2, 3],
    [3, 1],
    [1, 4],
    [2, 2]
])

# 使用numpy的argsort方法仅以第一列排序
sorted_indices = np.argsort(arr[:, 0])
sorted_arr = arr[sorted_indices]

print("123")
print(sorted_arr)

x= np.random.uniform(0.2,1.05)
print(x)

print(np.cos(x))
print(np.cos(1.01))