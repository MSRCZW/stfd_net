import pickle

# 定义.pkl文件的路径
pkl_file_path = "/home/zxy/dataset/kinect400/k400_hrnet.pkl"

try:
    # 打开.pkl文件以二进制读取模式
    with open(pkl_file_path, 'rb') as file:
        # 使用pickle.load()函数加载数据
        loaded_data = pickle.load(file)

    # 现在loaded_data中包含了.pkl文件中的数据
    # 在这里你可以对loaded_data进行任何你想要的操作

    # 例如，你可以打印loaded_data的内容
    print(loaded_data)

except FileNotFoundError:
    print(f"文件 {pkl_file_path} 未找到")

except Exception as e:
    print(f"读取文件时出现错误: {e}")
