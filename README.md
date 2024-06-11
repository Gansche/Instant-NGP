# Instant-NGP

Reproduction of paper Instant Neural Graphics Primitives

ZJU CG2023 Final Project


**Warning**

1. 服务器随时可能关停 每次写完务必上传GitHub (除了data/以外 已添加到gitignore)

2. conda 环境用 neus

   ```bash
   conda activate neus
   ```

3. debug 用cuda:3

    ```bash
    python ./scripts/debug.py
    ```

4. torch.version.cuda = 1.11.0+cu102

    其他environment待整理

**TODO List**

- [x] 摸清楚数据集结构 热狗模型的数据集 找两三个出来用
- [x] 其实可以先写dataset
    - 数据集写了能加载数据 但是跟没写一样，因为不知道后面咋用
- [x] 写主要的模型
- [x] 写loss
- [x] 调试以后写train
- [x] run!
- [ ] 自采数据集 colmap标定 需要读colmap
- [ ] loss更新
- [ ] lr更新


1. 第一阶段实现文章中提到的编码方式以及用一种数据集就可以了
2. 数据集就用热狗 好比较
3. 方法比较以及换数据集有空再说 没空就算

