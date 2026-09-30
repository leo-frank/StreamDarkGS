# Go2 + RealSense D435i 现场采集与重建

主流程使用接在 Go2 机载电脑上的 Intel RealSense D435i（序列号
`317622073182`）以 1280×720、15 FPS 拍摄彩色图。机器人通过 SSH 隧道
向本机传帧；本机持续评估候选帧，每秒选一张合格图连同相机内参送入
StreamDarkGS。当前 Pi3X 重建不使用深度，主流程关闭深度采集以保持候选帧率。
以下命令按终端执行，
**机器人终端使用 `/usr/bin/python3`，本机终端使用 bundle 内的 `env/bin/python`**。
Go2 内置 `/frontvideostream` 的旧流程在本文末尾。

## 机器人关闭后的 MP4 离线测试

这批 `20260507_171419` 数据已有彩色视频和同批次的 RealSense 彩色相机
标定文件。视频封装报告 **1280×720、30 FPS、5698 帧、约 189.93 秒**；
30 FPS 是 MP4 的时间基准，不能单凭它证明拍摄时相机一定逐帧以 30 FPS 输出。
同批次标定文件的 `K` 为 `fx=911.4570, fy=911.8998, cx=646.0638,
cy=372.5352`，畸变系数全零，与上面 D435i 在 1280×720 模式下的数值
一致。视频首帧与导出的原始 RGB PNG 都是 1280×720，画面对应。

在**本机**运行，不需要开启机器人、相机服务或 SSH 隧道：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
bash run_offline_realsense_mp4_pi3x.sh --prepare-only
bash run_offline_realsense_mp4_pi3x.sh --save-visuals
```

第一条只抽帧，便于检查；第二条重新抽帧并执行 Pi3X 重建。默认取视频前
60 秒，每个视频秒从约 30 帧里选细节分数最高的一张，约得到 60 张输入。
重建阶段会启动 `http://127.0.0.1:8765/profile?poll=50`，可在本机浏览器
查看最新输入图及随处理更新的 Albedo/Relit。抽帧准备阶段尚无预览；
重建和导出结束后服务自动关闭。`--save-visuals` 会另存 Viewer 预览图片。
这个离线筛选依据去噪后的图像细节，不使用录像中缺失的逐帧曝光/运动元数据，
因此不会保证每张都没有运动拖影。输出目录会打印为
`StreamDarkGS/output/offline_realsense_mp4_XXXXXXXX`，其中
`selection.jsonl` 记录原视频帧号，`input_frames/` 保存选中的 PNG 和同名
内参 JSON，`gaussian_map.pt` / `gaussian_map.ply` 是重建结果。
Pi3X 以 `--pi3x_intrinsics sidecar` 读取每帧的原尺寸内参；脚本会在图像
尺寸与标定不一致时直接报错。

需要处理全部视频时运行：

```bash
DURATION_SEC=0 bash run_offline_realsense_mp4_pi3x.sh --save-visuals
```

也可设置 `START_SEC`、`DURATION_SEC`、`OUTPUT_FPS`。例如只测试第 30 秒起
的 20 秒：`START_SEC=30 DURATION_SEC=20 bash run_offline_realsense_mp4_pi3x.sh`。
`REPLAY_FPS=0` 是默认值，表示准备完成后尽计算速度运行，不会按真实时钟
等待 1 秒再处理下一张；这只用于离线比较重建效果。

## 启动前检查

在**本机**确认 GPU、相机脚本和 Pi3X 权重都在：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
env/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
test -s models/pi3x/model.safetensors
test -f StreamDarkGS/tools/serve_realsense_rgbd.py
```

第一条应显示 `True` 和 GPU 名称。机器人和本机通过机器人 Wi-Fi 地址
`192.168.3.40` 通信；本流程不需要配置 Go2 内部 `eth1`、ROS 2 或
`/frontvideostream`。确认 D435i 已连接机器人，且旧的 Go2 H.264 源没有占用
机器人 `127.0.0.1:18800`。

## 1. 本机复制相机脚本

在**本机终端 A**运行，每次更新相机脚本后重新复制：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
scp StreamDarkGS/tools/serve_realsense_rgbd.py unitree@192.168.3.40:~/serve_realsense_rgbd.py
```

## 2. 机器人启动 D435i

若旧的相机脚本还在运行，先在其终端按 `Ctrl+C`；确认已完成第 1 步的脚本复制。
在**机器人终端**运行并保持这个进程运行。`--require-resolution` 保证本轮
不会自动回退到 640×480；固定 20 ms 彩色曝光和 4600 K 白平衡与最近的
现场测试保持一致：

```bash
/usr/bin/python3 -c 'import pyrealsense2, cv2, numpy; print("RealSense Python ready")'
/usr/bin/python3 ~/serve_realsense_rgbd.py \
  --width 1280 --height 720 --require-resolution \
  --camera-fps 15 --no-depth --color-exposure-ms 20 \
  --color-white-balance-k 4600
```

机器人 `base` 环境的 `python3` 是 3.14，之前不能导入 `pyrealsense2`；
系统 `/usr/bin/python3` 是 3.8，已经验证可以导入。成功启动会出现
`[realsense] selected ...` 和 `[realsense] listening on 127.0.0.1:18800`。
确认 `selected` 行包含 `color=1280x720@15` 和 `depth=None`；
这台设备之前已实际选中 15 FPS 纯彩色模式，并在本机筛选日志中测得约 15 FPS。
若仍显示 `@6` 或 `depth=(...)`，说明机器人端仍是旧命令，停止后按上面命令重启。
较早的带深度测试使用：

```text
[realsense] selected color=1280x720@6 format.yuyv depth=(480, 270)
```

该彩色模式的 SDK 内参实测为 `fx=911.4570`、`fy=911.8998`、
`cx=646.0638`、`cy=372.5352`，畸变系数全零。旧版脚本曾回退到：

```text
[realsense] selected color=640x480@6 format.bgr8 depth=(640, 480)
[realsense] serial=317622073182 ... depth_scale=0.001...
```

不带 `--no-depth` 时脚本才会尝试 RGB-D，包括带宽较低的 YUYV 彩色格式和
480×270 深度模式。此模式此前只能以约 6 FPS 稳定取得，
不适合下面的 15 FPS 整秒择优流程。
之前实测 640×480 RGB-D 的彩色内参为 `fx=607.6381`、
`fy=607.9332`、`cx=324.0425`、`cy=248.3568`，SDK 畸变系数均为零，
深度比例约 `0.001 m/单位`。以**本次运行打印的数值**为准。
主流程的 `depth=None` 是预期结果，本轮不会产生深度 PNG。

### 高分辨率模式与回退

第 2 步要求 1280×720 彩色图、15 FPS、无深度。看 `[realsense] selected ...`
确认实际模式，再按第 3 步做仅采集测试。检查生成 JSON 中
`color_intrinsics.width/height` 是否为 `1280/720`。
若该模式报 `Couldn't resolve requests` 且没有后续 `[realsense] selected ...`，
先检查 USB 连接和是否有别的进程占用相机。降到 640×480 需要重新评估图像
质量与内参，不要在本轮测试中静默回退。旧的 RGB-D 回退命令仅供单独对照：

```bash
/usr/bin/python3 ~/serve_realsense_rgbd.py --width 1280 --height 720
```

更高的 1920×1080 模式需使用相机支持的帧率（这台设备报告为 8 FPS），
并可能受 USB 链路限制；建议先确认 1280×720 稳定。

Pi3X 目前把输入缩放到约 25.5 万像素以内；提高相机分辨率能保留更大的
原始 JPEG 和更好的缩放前细节，但不会直接增加 Pi3X 的内部像素数。

### 记录曝光、减轻运动模糊

采集脚本现会在每帧 JSON 中记录 `color_actual_exposure_raw`（彩色帧曝光
元数据原值）、`color_gain_level`（彩色帧增益）和 `color_auto_exposure_frame`。
在这台 D435i 上，固定曝光 10 ms 时元数据返回 `100`，固定 20 ms 时返回
`200`，因此另存
`color_actual_exposure_ms_estimate = raw / 10` 供比较；这是根据本机实测
推断的换算，原值也会保留。早期采集中的 `color_actual_exposure_us` 字段
未经这个换算，不能直接按微秒解释。
若这台机器的帧元数据不可用，这三个字段为 `null`；同时记录的
`color_exposure_option_100us`、`color_gain_option` 和
`color_auto_exposure_option` 是**读取时的 SDK 选项值**，不能当作该帧的
实际曝光证明。新版脚本还记录 `color_white_balance_kelvin`（帧元数据）、
`color_white_balance_option_kelvin` 和 `color_auto_white_balance_option`；
设备不提供对应值时为 `null`。若要单独比较自动曝光，去掉第 2 步的固定曝光
参数重新采集；已有的旧采集没有这些字段。
本机可用下面的命令查看最近一张已保存图的曝光信息：

```bash
env/bin/python -c 'import glob,json; p=sorted(glob.glob("realsense_captures/*/frame_*.json"))[-1]; d=json.load(open(p)); print(p); print({k:v for k,v in d.items() if "exposure" in k or "gain" in k})'
```

固定 10 ms、增益 64 在当前办公室场景已实测明显偏暗（三帧平均灰度约
25/255）。固定 20 ms、同样增益的三帧平均灰度约 57–59/255，静止画面更清楚，
但仍偏暗；第 2 步以 20 ms 为完整重建测试的起始值。该选项关闭彩色自动曝光。

脚本会打印彩色曝光选项的设备支持范围；若设置不在范围内会报错，不会
悄悄使用其他值。移动测试至少采 10 帧（第 3 步命令改为 `--count 10`），
对比转向时的清晰度与亮度，再决定是否正式采集。若图像太暗，先增加照明；
也可在设备支持范围内加 `--color-gain 数值`
试验，但增益过高会增加噪声。取消 `--color-exposure-ms` 即恢复原来的自动曝光。
`--no-depth` 是本次主流程的配置；深度目前不参与 Pi3X 重建。关闭深度不会
直接缩短彩色曝光，但可让该设备使用已验证的 1280×720@15 彩色模式。

10 ms 的 30 帧移动测试中，第 27–29 帧颜色从偏青绿转为较中性；所有帧的
曝光原值均为 `100`、增益 `64`、彩色自动曝光关闭。第 26 到 27 帧的相机
时间戳相差约 15.5 秒。这个变化不能归因于曝光自动调整；更可能与白平衡
或现场光源色温变化有关。重新复制新版脚本后，白平衡字段可用于进一步区分。
另一组 10 ms 的 30 帧测试也从前半段偏青绿逐渐转为较中性，
`color_auto_white_balance_option=1`，但帧级白平衡元数据为 `null`；
选项值 `4600 K` 不能代表每帧实际色温。第 2 步固定为 4600 K 会关闭
彩色自动白平衡；这个值是已测试的起始值，需查看照片颜色再调整。
脚本会打印设备支持的白平衡范围。固定白平衡只能帮助验证和稳定颜色，不能
弥补曝光不足；正式采集仍应优先保证足够照明。

## 3. 本机建立 SSH 隧道

在**本机终端 B**运行并保持运行：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18800:127.0.0.1:18800 unitree@192.168.3.40
```

这个终端通常没有输出。本机 `127.0.0.1:18800` 此时连接到机器人的
RealSense 取帧程序。正式重建前，先在**本机另一个终端**执行
3 秒仅采集测试，不需要启动重建：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
env/bin/python StreamDarkGS/tools/send_realsense_rgbd.py \
  --capture-only --fps 1 --count 3 --selection motion \
  --candidate-fps 15 --selection-window-ms 1000 --max-motion-blur-px 3
```

测试文件保存在终端打印的 `realsense_captures/日期_时间/`；确认有
`events.jsonl` 和至少一对同名的 `frame_*.jpg`、`frame_*.json`。主流程不应有
`_depth.png`；每条事件的 `candidates` 应接近 15，结束时
`actual_candidate_fps` 应接近 15。以 JSON 中的 `color_intrinsics.width/height`
核对彩色图实际尺寸。若个别秒因全是模糊候选而跳过，首张文件编号可能不是
`000000`，查看 `events.jsonl` 的 `skipped` 原因。

## 4. 本机启动重建接收器

在**本机终端 C**运行，保持运行直到接收完成。下面的命令选择 Pi3X，
并把每帧 D435i 内参作为 Pi3X 条件输入：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
bash run_go2_local_bundle.sh --save-visuals --geometry-model pi3x --pi3x-intrinsics sidecar
```

等到 `[stream] waiting for frames` 再继续。终端会打印本轮独立的
`[output] StreamDarkGS/output/go2_local_XXXXXXXX` 目录。浏览器可打开
`http://127.0.0.1:8765/profile?poll=50` 查看输入图和重建预览。

若只想采集并使用原 Pi3 作基线，命令改为
`bash run_go2_local_bundle.sh --save-visuals`；原 Pi3 不使用 D435i 内参。

## 5. 本机发送正式重建输入（60 秒）

在**本机终端 D**运行：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
env/bin/python StreamDarkGS/tools/send_realsense_rgbd.py \
  --fps 1 --count 60 --selection motion --candidate-fps 15 \
  --selection-window-ms 1000 --max-motion-blur-px 3
```

本机持续从机器人获取彩色候选帧；每秒送出最多一张合格 JPEG 和对应相机 JSON，
同时将选中帧保存到 `realsense_captures/`。发送端逐帧打印 `server_frame`；
`--count 60` 表示 60 个一秒时间段，**不是保证发送 60 张**。若某秒全部候选
不合格，该秒跳过并在 `rejected/` 保存一张供复核。至少送出一张后，时间段
结束时自动发 `/finish`；等终端 C 打印 `[done]` 才算重建结束。
若先用 `--count 10` 测试了重建，正式采集前需重新启动一个全新的终端 C
接收器；一轮不能先送 10 秒再续送 60 秒。

### 整秒持续筛选规则

第 2、5 步是完整重建的当前推荐配置。发送端后台持续取帧，
追踪相邻帧中的静态场景特征，并结合彩色帧的实际曝光
时间估算曝光期间的画面位移（以原始彩色图像素计）。在每个 1 秒时间段内
持续评估约 15 张图：出现合格图就暂存，后面有更清晰的合格图就替换；
时间段结束时只发送最清晰的一张，随后开始下一秒。若该秒全部候选都不合格，
则跳过这一秒，并将最清晰的被拒图保存在 `rejected/`。选择只看本时间段，
不会重复发送上一秒的图。`--selection motion --fps 1` 默认使用完整 1000 ms
时间段，命令中显式写出以便核对。

合格判断先看估计拖影是否不超过 3 像素；即使超过，图像细节仍保有最近清楚帧
至少 70% 时也保留。只有估计运动偏大且图像细节也明显下降时才跳过。
如果特征不足或曝光元数据缺失，运动估计为未知；这时仅在它的清晰度
低于最近 1.5 秒可靠低运动帧中位数的 40% 时跳过。这是针对跟踪失败时
明显拖影的补救规则，不使用固定的全图清晰度阈值。
亮度只记录在日志里，不参与筛选；清晰度用于同一秒内择优和相对细节下降
判断，但不再使用固定的 `--min-sharpness 20` 门槛。
旧命令中的 `--min-brightness` 现在会被忽略，请从命令中删除。
`--fps 1` 控制送入重建的最高帧率，`--candidate-fps 15` 控制相机取帧目标速率；
实际取帧速率取决于相机、JPEG 编码和 SSH 链路，可看结束时的
`actual_candidate_fps`。`events.jsonl` 逐次记录 `candidates` 和
`candidate_frame_numbers`，可确认每秒实际比较的图片；还记录 `camera_read_ms`、`score_ms`、
`motion_ms`、`motion_blur_px`、`motion_features`、
`motion_sharpness_ratio`、`motion_fallback_blurry`、`decision_ms` 和选中帧的
`selected_age_ms`；如有上传还记录 `upload_ms`。
`--max-motion-blur-px 3`、`--motion-sharpness-ratio 0.7` 和
`--motion-fallback-ratio 0.4` 是待现场校准的
起始阈值；先用
`--capture-only --count 10` 对照保存图与 `rejected/` 看误判，再调整该值。
相邻帧位移只能估计运动拖影，不能直接测出这张图的真实模糊程度；突然启停、
滚动快门、大幅运动导致的特征跟踪失败和突然切换到低纹理场景仍可能误判或漏判。
若需保留旧的全图清晰度评分方法，使用 `--selection sharpest --min-sharpness 20`。
完整 1000 ms 时间段意味着选中的图片可能已拍摄将近 1 秒；若重建更重视
图像时效，可用 `--selection-window-ms 500` 缩短选取范围，但每秒可比较的
图片也会降至约 7–8 张。

上面的 `--no-depth` 只用于当前 Pi3X 彩色输入，可减少每个候选的传输和处理量；
如需存档深度，去掉它后确认机器人实际选中的彩色与深度模式，以及本机
`actual_candidate_fps`。缩成 640×480 不会消除已经由曝光和机身运动造成的模糊；
如果 1280×720 模式的实际候选速率不足，再单独比较 640×480 模式。

## 文件位置与实际使用的数据

```text
realsense_captures/日期_时间/
  frame_000000.jpg           选中帧的彩色 JPEG（跳过的秒可能使编号不连续）
  frame_000000.json          该帧 SDK 内参、序列号和时间戳
  frame_000000_depth.png     对齐深度，仅单独启用 RGB-D 时出现
  rejected/window_*.jpg     全部不合格的秒中最清晰的一张，供人工复核
  session.json              本轮筛选参数
  events.jsonl              每秒的候选帧号、跳过记录及接收器返回值

StreamDarkGS/output/go2_local_XXXXXXXX/
  stream_capture_日期_时间/
    stream_...jpg            接收器实际用于重建的彩色图
    stream_...json           与 JPEG 同名的相机 JSON
  gaussian_map.pt            重建地图
  gaussian_map.ply           可查看的点云/Gaussian 导出
  cameras.json              重建相机参数
  run.log                   完整运行日志
  viewer_frames/             使用 --save-visuals 时的预览图
```

深度 PNG 的像素值乘 JSON 中 `depth_scale_m_per_unit` 得到米。当前 Pi3X
`--pi3x-intrinsics sidecar` 会按模型输入大小缩放 SDK 内参并传入模型，同时用于
该帧融合相机；**深度 PNG 目前只归档，没有送入 Pi3X，也没有参与几何融合**。
缺少相机 JSON、图像尺寸不符或畸变系数非零时，`sidecar` 会报错而不是悄悄
退回估计内参。

## 常见问题

- `ModuleNotFoundError: pyrealsense2`：机器人端使用 `/usr/bin/python3`。
- `cv2.cvtColor ... Invalid number of channels ... scn is 1`：机器人上仍是旧版
  YUYV 解码脚本。停止该进程，按第 1 步重新复制脚本，再按第 2 步启动。
  此错误也会让本机发送端报 `camera stream closed after 0/12 bytes`；重启后
  先完成第 3 步的 3 帧测试，再启动正式采集。
- `Couldn't resolve requests`：单个相机 profile 不可用时会自动试下一组；
  看到后续 `[realsense] selected ...` 就表示已成功。
- `Address already in use`：确认机器人上旧的 Go2 H.264 服务没有占用 18800。
- `Receiver is not fresh`：上一轮接收器已收到图像；重新运行第 4 步。
- `camera stream closed` 或发送中断：保留已保存的 `realsense_captures/`；
  检查第 2、3 步，重新启动第 4 步的新接收器后再发正式采集。

## 模型选择

`run_go2_local_bundle.sh` 默认使用 Pi3。Pi3X 源码在 `Pi3X-runtime/`，权重
在 `models/pi3x/model.safetensors`。仅用 RGB、让 Pi3X 自行推断内参时运行：

```bash
bash run_go2_local_bundle.sh --save-visuals --geometry-model pi3x
```

已知 D435i 内参时使用第 4 步的 `--pi3x-intrinsics sidecar`。这个选项是真正的
模型条件输入，不是只替换输出中的焦距数值。
[Pi3X 官方模型卡](https://huggingface.co/yyfz233/Pi3X)把权重标为 CC BY-NC 4.0（非商业使用）。

## Go2 内置相机旧流程

这份文档记录本机已跑通的流程。GO2机载电脑订阅ROS 2话题`/frontvideostream`，
通过SSH隧道把H.264视频送到本机；本机从最近画面中选清晰帧，以1 FPS送入
StreamDarkGS重建。以下命令使用当前机器的实际路径与GO2 Wi-Fi地址`192.168.3.40`。

## 启动前确认

在本机运行：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
env/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
test -x /home/igrape/anaconda3/envs/ros1/bin/ffmpeg
```

应输出`True NVIDIA GeForce RTX 3090`，且`test`命令没有报错。
环境已经安装在包内的`env/`，日常运行不需要进入其他Conda环境，也不使用
`go2-sdk-env`的`VideoClient.GetImageSample()`取图。

## 1. 本机终端：复制视频源并登录机器人

每次更新`serve_go2_ros_h264.py`后先复制；脚本已经是最新版时可以跳过`scp`。

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
scp StreamDarkGS/tools/serve_go2_ros_h264.py unitree@192.168.3.40:~/serve_go2_ros_h264.py
ssh unitree@192.168.3.40
```

以下第2步在登录后的**机器人终端**执行。

## 2. 机器人终端：恢复内部网卡并启动ROS 2视频源

`eth1`即使显示`UP,LOWER_UP`，也可能在重启后丢失`192.168.123.18/24`。
先执行下面整段命令；地址已经存在时不会重复添加。

```bash
ip -br link show dev eth1
ip -4 addr show dev eth1 | grep -q 'inet 192.168.123.18/24' || sudo ip addr add 192.168.123.18/24 dev eth1
ip -4 addr show dev eth1
ip route get 239.255.0.1 oif eth1
ping -I eth1 -c 3 192.168.123.161
```

确认路由显示`dev eth1 src 192.168.123.18`，且`ping`收到回复。
若路由仍显示`src 172.17.0.1`，这是Docker地址，说明机器人内部网络尚未恢复，
此时不要继续启动视频源或发送帧。

网络检查通过后，在同一机器人终端运行并保持运行：

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>eth1</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
/usr/bin/python3 ~/serve_go2_ros_h264.py
```

必须看到`[source] video720p=...`的数字持续增长，才继续下一步。
`[source] listening on robot 127.0.0.1:18800`只表示端口开启；
`client connected`也只表示客户端连上，并不代表视频正在更新。
`NetworkInterfaceAddress: deprecated element`只是配置提示；
若出现`ddsi_udp_conn_write to udp/239.255.0.1:7400 failed`，按`Ctrl+C`停止视频源，
重新检查上述`eth1`地址、路由与`ping`，修复后再启动。

### 可选：从本机控制机器狗头部灯

GO2 VUI亮度接口使用`0`关灯、`1`到`10`设置亮度。在**本机新终端**执行：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
bash go2_light.sh status
bash go2_light.sh on 8
```

开灯后需要关灯时，单独执行：

```bash
bash go2_light.sh off
```

`bash go2_light.sh toggle`可切换状态。命令通过SSH把当前灯光脚本送到机器人执行，
无需手动复制脚本，也不会影响正在运行的视频源。灯光脚本使用机载ROS 2的
`/api/vui/request`接口，避开此前会导致段错误的Python SDK初始化。
成功后会读回亮度并显示`brightness=8 (on)`或`brightness=0 (off)`；
实际对场景的补光效果以相机画面为准。

## 3. 本机终端B：保持SSH视频隧道

新开一个**本机**终端，运行并保持运行：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18800:127.0.0.1:18800 unitree@192.168.3.40
```

这个命令正常运行时通常没有输出。本机`127.0.0.1:18800`现在通向机器人视频源。

## 4. 本机终端A：启动重建接收器

再开一个**本机**终端：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
bash run_go2_local_bundle.sh --save-visuals
```

等到`[stream] waiting for frames`。在本机浏览器打开：

```text
http://127.0.0.1:8765/profile?poll=50
```

页面并排显示最新送入重建的输入图、Albedo和Relit。输入图收到后立即刷新，
重建图随算法结果刷新。`--save-visuals`还会保存Viewer发布的画面。

## 5. 本机终端C：发送视频中的清晰帧

再开一个**本机**终端，运行以下60张、1 FPS的正式采集命令：

```bash
cd /media/igrape/fanwg_1T/zxn_go2/go2_local_bundle
env/bin/python StreamDarkGS/tools/send_go2_ros_stream.py \
  --ffmpeg /home/igrape/anaconda3/envs/ros1/bin/ffmpeg \
  --fps 1 --count 60 --selection sharpest --sharpness-window-ms 200
```

发送端持续接收约15 FPS的原始H.264视频并保存为`go2_captures_ros/日期_时间/camera.h264`；
每秒只从最近200毫秒内**已解码**的帧中选一张较清楚的JPEG送往重建，不额外等待凑帧。
`acknowledged`表示这一张已被接收器接受。完成60张后发送端自动调用`/finish`；
等终端A打印`[done]`，再查看输出。

如果只想先测试10张，把`--count 60`改成`--count 10`。
测试结束后必须重新运行第4步，启动一个**新接收器**，再执行正式采集；
同一接收器会话不能先发10张再继续发60张。

## 视频中断时的恢复

若发送端报`No fresh decoded frame`，先看机器人终端的`video720p`数字是否仍在增长。
这次现场故障的原因是机器人`eth1`没有IPv4地址：DDS组播错误地用了
`172.17.0.1`，发送端只收到最初几帧。返回第2步恢复地址和视频源。
接收器如果已经接受了部分图片，本轮不能续传；视频恢复后，重新运行第4步和第5步。
发送失败时已保存的`camera.h264`、JPEG和日志仍在对应的`go2_captures_ros/`目录。

`--fps 1`只控制送入重建的帧率，不改变原始视频流。
`--sharpness-window-ms 100`可以缩短选中帧的回看时间，但候选帧更少；
`--selection latest`可直接发送最新画面。清晰度择优无法在所有候选帧都模糊时产生清晰图。
当前实测重建能跟上1 FPS；提高帧率后需要重新检查队列和Viewer延迟。

## 结果位置

```text
StreamDarkGS/output/go2_local_XXXXXXXX/
  gaussian_map.pt
  gaussian_map.ply
  run.log
  summary.csv
  module_times.csv
  viewer_frames/       使用--save-visuals时存在
  stream_capture_*/    接收器实际收到的输入图

go2_captures_ros/日期_时间/
  camera.h264           原始视频流
  frame_*.jpg           选中并发送的图片
  events.jsonl          逐帧发送记录
  summary.json          本次发送结果
```

`summary.csv`和`module_times.csv`用于分析处理速度；`events.jsonl`含每张图的
候选帧数量、清晰度分数及选中帧的解码后时长。
