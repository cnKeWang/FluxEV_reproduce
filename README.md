# FluxEV_reproduce

data_preprocess是预处理，测试选取的时序数据没有丢失值，故只做了读取csv数据

fluctuation_extraction是波动提取和两步平滑。对应算法1

pot是peaks-over-threshold。对应算法2。
grimshaw是矩量法，返回均值和方差。是POT算法的其中一步。

main是算法3。
smooth是平滑操作()。对应算法3中的CalsFeats,这个不太确定是不是写对了。文中说是(1)-(7)式
