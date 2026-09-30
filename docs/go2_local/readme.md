# GO2现场本机重建：打包、搬运与运行

本方案把重建算法和模型带到现场Ubuntu电脑。GO2图像只经过现场局域网进入该电脑，
Pi3、MVInverse、SAM2、高斯融合、在线优化和Viewer全部在现场执行，不需要VPN、SSH隧道或远程GPU。

## 1. 先确认现场电脑合格

现场电脑必须是x86-64 Ubuntu，并配有NVIDIA GPU。当前实测运行峰值为：

- GPU已分配显存约11.5 GB，保留显存约12.9 GB；最低建议16 GB，推荐24 GB。
- 当前环境是Python 3.10、PyTorch 2.5.1+cu118；驱动必须能运行CUDA 11.8程序。
- 一次48帧实验输出约675 MB；建议至少预留100 GB磁盘。
- 完整离线包解压后约18 GB以上，U盘/移动SSD建议至少32 GB，现场磁盘建议40 GB以上。

到现场前让对方执行并保存结果：

```bash
uname -m
lsb_release -a
nvidia-smi
df -h
```

没有NVIDIA GPU、显存不足或`nvidia-smi`报错时，这套配置不能在现场实时运行。

## 2. 必须打包的内容

| 内容 | 作用 | 大致未压缩大小 |
|---|---|---:|
| StreamDarkGS运行代码 | 流水线、Viewer、计时和结果保存 | 数MB，不含output |
| Pi3最小运行源码 | 几何、深度和位姿预测 | 约2 MB |
| Pi3真实权重 | Pi3模型 | 约3.6 GB |
| MVInverse源码和真实权重 | 材质预测 | 权重约3.8 GB |
| SAM2源码和base-plus权重 | 分割和缝隙补全 | 权重约309 MB |
| gsplat源码与已编译扩展 | 高斯融合、优化、渲染 | 约36 MB |
| 当前Python环境 | 保持与已验证实验一致 | 未压缩约9.5 GB |
| GO2采集脚本 | 从相机取图并发往本机 | 已包含在tools目录 |

不要复制整个`StreamDarkGS`或`Pi3-main`目录：它们当前分别约72 GB和242 GB，绝大部分是旧输出与数据。
也不要普通复制Hugging Face snapshot中的权重链接；Pi3和MVInverse权重是符号链接，必须复制实际文件。

## 3. 在当前GPU电脑生成离线包

准备一个不存在的新目录，最好位于空间充足的移动SSD：

```bash
cd /home/zxn/disk/StreamDarkGS
bash tools/prepare_go2_local_bundle.sh /绝对路径/go2_local_bundle
```

脚本会：

1. 排除旧`output`、日志、Git元数据和大数据集。
2. 复制Pi3、SAM2和gsplat所需源码。
3. 使用`cp -L`复制三套真实权重，不保留失效的缓存符号链接。
4. 使用`conda-pack`打包当前已验证环境。
5. 生成`SHA256SUMS`用于现场校验。

该步骤会读取约18 GB数据并压缩环境，可能需要较长时间。脚本拒绝覆盖已有目标目录。
完成后复制整个`go2_local_bundle`目录，不能只复制README或运行脚本。

复制完成后在源机器或移动SSD上检查：

```bash
cd /绝对路径/go2_local_bundle
sha256sum -c SHA256SUMS
du -sh .
```

## 4. 现场Ubuntu解压环境

把整个包复制到现场本地SSD，例如`/data/go2_local_bundle`。不要直接从慢速U盘运行。

```bash
cd /data/go2_local_bundle
sha256sum -c SHA256SUMS
mkdir -p env
tar -xzf mvinverse-gsplat-cu118.tar.gz -C env
env/bin/conda-unpack
```

打包环境里原有SAM2/gsplat是editable安装并含旧机器绝对路径；本方案运行脚本通过`PYTHONPATH`
显式使用包内源码和已编译扩展，因此不需要联网重装，也不要删除`sam2-runtime`和`gsplat-runtime`。

先做离线检查：

```bash
cd /data/go2_local_bundle
PYTHONPATH="$PWD/sam2-runtime:$PWD/gsplat-runtime" env/bin/python - <<'PY'
import torch, cv2, sam2, gsplat
print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('sam2:', sam2.__file__)
print('gsplat:', gsplat.__file__)
PY
```

必须看到`cuda available: True`，且SAM2、gsplat路径位于当前包中。

## 5. 现场连接结构

```text
GO2自带相机
    │ GO2现场局域网
    ▼
现场Ubuntu：宇树SDK取图进程
    │ http://127.0.0.1:8765
    ▼
同一台Ubuntu：Pi3 → MVInverse → SAM2 → 高斯融合/在线优化
    │
    ▼
本机浏览器Viewer + 本地output结果
```

建议使用两个Python环境和两个终端：

- 终端A使用离线包中的算法环境。
- 终端B使用现场已经安装并验证过宇树SDK的环境。

不必把宇树SDK安装到算法环境，也不再使用SSH端口转发。

## 6. 终端A：启动本机重建

```bash
cd /data/go2_local_bundle
bash run_go2_local_bundle.sh --save-visuals
```

看到下面两项后保持终端运行：

```text
[output] .../StreamDarkGS/output/go2_local_XXXXXXXX
[viewer] http://127.0.0.1:8765/profile?poll=50
```

在现场Ubuntu浏览器打开：

```text
http://127.0.0.1:8765/profile?poll=50
```

`--save-visuals`会保存Viewer实际发布的albedo和relit画面；不需要保存时可省略。

## 7. 终端B：从GO2相机发送到本机

激活现场已经能运行宇树Python SDK的环境，确认GO2网卡名：

```bash
ip -br addr
export GO2_IFACE='实际GO2网卡名'
```

先静止发送10张测试：

```bash
cd /data/go2_local_bundle/StreamDarkGS
python tools/send_go2_frames.py \
  --interface "$GO2_IFACE" \
  --server http://127.0.0.1:8765 \
  --fps 1 --count 10 \
  --output-dir /data/go2_capture_test
```

确认Viewer正常更新后再正式采集，例如60张：

```bash
python tools/send_go2_frames.py \
  --interface "$GO2_IFACE" \
  --server http://127.0.0.1:8765 \
  --fps 1 --count 60 \
  --output-dir /data/go2_capture_$(date +%Y%m%d_%H%M%S)
```

发送完成后脚本会通知接收器结束，算法完成尾部在线优化、结果导出和计时汇总。
不要在同一个接收器会话里先发10张再继续发60张；测试结束后等终端A完成，再重新启动终端A正式运行。

## 8. 结果和计时

每次运行结果位于：

```text
StreamDarkGS/output/go2_local_XXXXXXXX/
  gaussian_map.pt
  gaussian_map.ply
  run.log
  summary.csv
  module_times.csv
  viewer_frames/       使用--save-visuals时存在
  stream_capture_*/    GPU端实际接收到的输入
```

现场采集脚本自己的原图和发送日志位于`--output-dir`。实验结束后应同时带回：

- 整个`go2_local_XXXXXXXX`结果目录；
- 现场采集目录；
- 如有Viewer录屏，也带回浏览器下载的视频。

## 9. 出发前必须完成一次同机演练

不要等到现场才第一次解包。应在另一台满足条件的Ubuntu GPU电脑上完成：

1. `sha256sum -c SHA256SUMS`全部通过。
2. 环境解包和CUDA检查通过。
3. 启动`run_go2_local_bundle.sh`。
4. 使用`send_go2_frames.py --frames-dir 某个测试图片目录`模拟GO2输入。
5. Viewer可见、程序正常结束，生成`summary.csv`与`module_times.csv`。

已编译的gsplat和SAM2扩展依赖x86-64、当前PyTorch/CUDA ABI及兼容NVIDIA驱动。
如果现场GPU架构或驱动差异很大，必须提前在同型号机器演练；仅仅“有CUDA”并不足以保证可运行。
