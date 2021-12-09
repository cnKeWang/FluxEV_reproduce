# FluxEV_reproduce
demo只是对数据做了两步平滑操作的可视化

fluctuation_extraction.py 和smooth.py都是两步平滑，没有本质区别，一个是对初始阈值
用到的数据做平滑，另一个是后面所有数据更新动态阈值做平滑。参考算法3

同上，pot.py 和 poty.py 也是考虑前后数据处理略有不同。

需要调的参数有smooth.py中第13行alpha和fluctuation_extraction.py中31行的alpha, 这是paper中没有给出的
参数，还有一些参数我针对特定的数据集调了发现有效果如 window size k = 150, risk coefficient q = 1e-4(这个好像对结果影响还挺大的)以及初始阈值t的init_level

理论上是复现了，precision和recall有一个写错了。
