# FluxEV_reproduce

RUN demo:

demo中的data_path为数据路径， 读取数据value。

firstdemo()先对数据做提取和两步平滑，处理后得到的E, f, m, S(定义见fluctuation_extraction.py)。
然后作图画出初始数据value和E, f, m, S

需要调的参数有smooth.py中第8行alpha和fluctuation_extraction.py中31行的alpha