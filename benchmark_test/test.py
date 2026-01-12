import os
benchmark_files = []
for file in os.listdir('.'):
    if file.startswith('benchmark') and os.path.isfile(file):
        benchmark_files.append(file)
benchmark_files.sort()  # 按名称排序
print(benchmark_files)