# GO2 EDU → 远程GPU重建 → 现场Ubuntu实时观看

本手册适用于：GO2 EDU自带前置相机、现场Ubuntu已安装宇树SDK、现场可SSH登录GPU电脑。
GPU项目位置：`/home/zxn/disk/StreamDarkGS`。

**先静止验证，再遥控慢走。这里的脚本只读取相机，不控制机器狗运动。**
取图接口已按宇树官方Python示例核对；真实GO2取流、现场网络和行走质量尚待现场验证。
已验证的手机/视频耗时不保证适用于GO2的画幅和网络。

## 1. 最终链路和运行位置

```text
GO2 EDU前置相机
    │ 现场机器人网络（SDK）
    ▼
现场Ubuntu：相机取图 → 选中图片落盘 → HTTP上传
    │ http://127.0.0.1:18765
    │ SSH加密隧道
    ▼
GPU电脑：127.0.0.1:8765 → Pi3 → MVInverse → 材质处理 → 高斯融合
                                                    ├─ 在线优化
                                                    └─ 渲染、编码
    │ 同一SSH隧道返回
    ▼
现场Ubuntu浏览器：http://127.0.0.1:18765/profile?poll=50
```

两个`127.0.0.1`指各自电脑，不是GO2地址！GO2局域网地址也不是GPU电脑的SSH地址。
机器人SDK流量留在现场，只有选中的图片和预览通过SSH传输。

本轮默认：W3/S2、目标1 FPS、create stride=2、每次在线优化10步、低延迟调度、非创建帧预览。
Pi3与MVInverse串行；在线优化可与下一窗口预测重叠。已触发的优化不跳过。
最终离线优化关闭，避免首次实验结束后再等待大量离线迭代。

## 2. 出发前检查表

- 确认现场有Ubuntu电脑、机器人网络连接方式和互联网连接。
- 现场电脑必须同时能访问GO2和GPU电脑。连接机器人热点可能改变默认路由、导致互联网断开。
  优先使用现场已有的可靠网络配置，不要远程盲改网卡IP、默认路由或关闭防火墙。
- 带好SSH登录信息；如需跳板机、非22端口，沿用平时能成功登录的参数。
- GPU电脑保持开机、模型文件可用；实验时不要并行运行其他GPU任务。
- 现场准备遥控器、充足电量和安全测试区域。重建Viewer不能当作安全避障或遥控反馈。
- 现场SDK环境应能导入`unitree_sdk2py`、`cv2`、`numpy`，不需要安装本项目全部GPU依赖。

本次提供三个文件：

- 本手册：`docs/go2_remote/readme.md`
- 现场发送端：`tools/send_go2_frames.py`
- GPU接收端启动脚本：`tools/run_go2_receiver.sh`

## 3. 现场Ubuntu：下载发送脚本和手册

打开现场终端，设置变量。把示例值换成真实信息；`GPU_SSH`也可以填已有的SSH配置别名。

```bash
export GPU_SSH='你的用户名@GPU电脑可访问的地址'
mkdir -p ~/go2_remote_demo
cd ~/go2_remote_demo
scp "${GPU_SSH}:/home/zxn/disk/StreamDarkGS/tools/send_go2_frames.py" .
scp "${GPU_SSH}:/home/zxn/disk/StreamDarkGS/docs/go2_remote/readme.md" .
```

若SSH使用非默认端口，`ssh`用`-p 端口`，`scp`用`-P 端口`；以下命令都要对应调整。
不要把登录密码写进脚本。首次连接应核对服务器指纹。

激活现场已安装宇树SDK的环境，再检查：

```bash
python -c "import sys; print(sys.executable); import cv2, numpy; from unitree_sdk2py.go2.video.video_client import VideoClient; print('SDK imports OK')"
python send_go2_frames.py --help
```

若只有C++ SDK，或Python导入失败，先按官方Python SDK安装说明配置Python环境。
不要为此重装GPU电脑环境。官方链接见文末。

## 4. 现场Ubuntu：先验证GO2相机，暂不上传

### 4.1 找到连接机器狗的网卡

```bash
ip -br addr
ip route
```

请现场同事确认哪块网卡连GO2，然后设置（下例必须替换）：

```bash
export GO2_IFACE='enp3s0'
```

网卡可能是以太网或无线网卡；不要默认选VPN、Docker或互联网网卡。
机器人的具体IP、子网和SDK配置沿用现场已验证的设置，本文不假定固定IP。
仅能ping通不代表SDK相机服务一定可用。

### 4.2 无图形界面也能做的取图测试

让机器狗先静止，并在SDK Python环境中执行：

```bash
cd ~/go2_remote_demo
python send_go2_frames.py \
  --interface "$GO2_IFACE" \
  --capture-only --fps 1 --count 3
```

成功时出现`[local-output] ...`、三条`saved`和一条`complete`。
通过现场文件管理器打开打印目录中的三张JPG，检查：

- 确实是自带相机的当前画面，不是历史图片或重复冻结帧。
- 图像方向、尺寸、清晰度正常；没有严重黑屏、过曝或明显畸变问题。
- 不要直接拉伸成手机的竖屏尺寸。发送脚本保持解码后的原始尺寸。

`--capture-only`不访问GPU电脑，也不需要SSH隧道。
如果`GetImageSample`返回非零错误码，先解决现场SDK/网络/相机服务问题，不要继续后面的重建。
不要运行运动控制、低层电机或障碍开关示例来“验证相机”。

## 5. 终端A：登录GPU电脑，启动接收与重建

在现场新开一个终端，用平时能登录的方式：

```bash
ssh "$GPU_SSH"
```

**下面两条在SSH登录后的GPU电脑上执行，不是在现场本机执行：**

```bash
cd /home/zxn/disk/StreamDarkGS
bash tools/run_go2_receiver.sh
```

模型加载需要等待。看到`[stream] waiting for frames`后再启动上传。
记下最先打印的`[output] /home/zxn/disk/StreamDarkGS/output/go2_live_XXXXXXXX`。
每次启动都新建输出目录，不覆盖旧实验。

保持终端A运行。建议使用已有的tmux/screen会话，防止SSH意外断开杀死进程。
若已安装tmux，可先运行`tmux new -s go2-live`再启动脚本；`Ctrl+B`后按`D`分离，
重连后`tmux attach -t go2-live`查看。不要按Ctrl+C结束尚未收尾的任务。

接收程序**只监听GPU电脑的127.0.0.1:8765**，不需要开放公网8765端口。
启动脚本使用本项目当前Python环境和模型路径，若迁移GPU电脑，需先调整脚本中的路径。

## 6. 终端B：现场Ubuntu建立SSH隧道

这是现场本机终端，不要在已经SSH登录GPU电脑的终端里执行。
新终端要重新设置`GPU_SSH`（终端之间不会自动共享刚才export的变量）：

```bash
export GPU_SSH='你的用户名@GPU电脑可访问的地址'
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18765:127.0.0.1:8765 \
  "$GPU_SSH"
```

正常情况下没有输出，终端一直占用。保持它运行。
SSH配置必须允许TCP端口转发；能SSH登录不一定代表管理员允许转发。
出现`administratively prohibited`时请管理员确认，不要尝试绕过策略。

## 7. 终端C与浏览器：验证连接、打开重建画面

在现场另开终端C：

```bash
curl --max-time 5 http://127.0.0.1:18765/status
```

应返回JSON，首次上传前应为`received: 0`、`finished: false`。
如果不是空会话，先确认是否还有其他发送程序；需要新实验时在GPU端重新启动接收程序。

在**现场Ubuntu的浏览器**打开：

```text
http://127.0.0.1:18765/profile?poll=50
```

保持页面在前台。左图为高斯地图的Albedo渲染，右图为重光照结果，不是直接显示相机视频。
尚未送图时显示“等待服务器预览”是正常的。不要打开根页面并点击手机采集，避免两种输入混入同一会话。
如果同时多开计时页面，会产生重复的浏览器确认记录；测试只保留一个页面。

## 8. 第一次端到端实验：静止或小幅慢移，发送10张

现场终端C激活SDK Python环境，重新设置网卡变量：

```bash
cd ~/go2_remote_demo
export GO2_IFACE='enp3s0'
python send_go2_frames.py \
  --interface "$GO2_IFACE" \
  --server http://127.0.0.1:18765 \
  --fps 1 --count 10
```

建议先保持静止确认传输，之后小幅、缓慢平移，避免纯静止数据视差不足导致重建质量不好。

发送端正常输出：

1. `saved`：本地图片已保存。
2. `upload_started`：开始上传。
3. `acknowledged`：后端明确确认接收，包含后端图片名。
4. 全部10张确认后，自动发送`/finish`，最后输出`complete`。

GPU端应该依次看到：`stream-timing`、`window`、`fusion`、`viewer-submit`、
`online-global-opt`。浏览器绘制成功后还会出现`viewer-client-timing`。
首个窗口需要收齐3张图，再做模型推理；不要期待第一张上传后立即显示。

发送结束不等于计算结束！等待GPU端出现`[save]`和`[done]`，然后检查输出文件。
本轮结束后接收进程会退出，网页后续可能提示无法连接，这不代表保存失败。

## 9. 正式实验：重新启动接收，慢走采集60张

每个实验使用新的接收会话。终端A在上一轮完成后再次运行：

```bash
cd /home/zxn/disk/StreamDarkGS
bash tools/run_go2_receiver.sh
```

终端B隧道可继续使用；若断开则重建。确认`/status`清零，刷新并保持Viewer前台。

现场终端C：

```bash
cd ~/go2_remote_demo
python send_go2_frames.py \
  --interface "$GO2_IFACE" \
  --server http://127.0.0.1:18765 \
  --fps 1 --count 60
```

建议有人用遥控器负责行走，另一人看原图与Viewer。先缓慢直行，再缓慢转弯，最后回看已走过的区域。
避免急转、快跑、近距离扫过墙面及人群；不要边盯延迟Viewer边用它代替现场观察。
低视角遮挡和步态模糊可能影响Pi3，需检查回访区域是否错位，不只看画面是否流畅。

### 当前发送器的重要限制

- `--fps 1`是目标采样频率，`--count 60`是成功取图/发送数量，不是严格60秒计时器。
- 首版采用“取图→本地保存→上传确认”串行流程，网络或取图慢会降低实际FPS，不突发补发过期图片。
- 保存的是选中的图像，不是相机全帧率视频，也不承诺图像间严格一秒。
- 相机数据解码后按原尺寸编码为质量95的JPEG；不是无损原始传感器数据。
- 记录的是取图完成时间，不是相机曝光时间。相机SDK/网络可能已有缓冲，需现场检查。
- 创建帧仍按create stride=2选取；其他帧只预览，不增加优化次数。
- 非创建帧预览仍等待位姿预测；没有位姿插值，也不是10/30 FPS播放。
- Viewer最新优先，可能合并来不及渲染的旧预览；这不等于跳过已触发的在线优化。

## 10. 数据在哪里？如何判断完成？

### 最终实验A：统计模块耗时与真实显示延迟

使用第9节的默认启动命令，不额外保存渲染图片、不录屏，以减少测量干扰：

```bash
# GPU电脑，终端A
cd /home/zxn/disk/StreamDarkGS
bash tools/run_go2_receiver.sh
```

现场按第9节发送60张图；Viewer保持前台，确认GPU日志出现`viewer-client-timing`。
正常处理并保存完成后，会自动在本轮`output/go2_live_XXXXXXXX/`生成：

- `summary.csv`：完成状态、输入帧数、创建/预览次数、显存、整体时间。
- `module_times.csv`：Pi3、MVInverse、SAM、补缝、创建、融合、后台优化、渲染及浏览器ACK的均值/P50/P95/最大值。
- `case.json`：启动配置（本脚本固定W3/S2、create stride=2、目标1 FPS）。

重点查看：

| 指标 | 含义 |
|---|---|
| `window-timing / total_ms` | 窗口预测阶段，不是完整窗口完成延迟 |
| `profile-online-worker / online_optimization_ms` | 后台优化实际执行时间；不能与重叠模型时间重复相加 |
| `optimization-wait / wait_ms` | 主线程等优化收尾的时间 |
| `profile-viewer / source_age_ms` | 后端接收源图到渲染发布，不含浏览器显示 |
| `viewer-client-timing / receive_to_ack_ms` | 接收到浏览器绘制确认返回的近似显示延迟上界 |
| `receive_to_done_ms` | 第一张到达后到全部处理完成 |

`processing_ms`包含服务启动后等待你开始送图的空闲时间，不能直接比较不同轮次的纯处理速度。
`early/middle/late`按输入序号和目标1 FPS划分，不是严格的实际墙钟时间段。
汇总里的`complete`主要依据处理与保存日志，优化是否全做完还需核对优化历史。

如果进程异常退出，自动汇总未执行，可在GPU电脑手动汇总已有日志（先替换目录）：

```bash
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python \
  tools/summarize_go2_run.py --output-dir output/go2_live_XXXXXXXX
```

这个汇总入口只适用于配套接收脚本的固定参数；手动修改窗口/FPS后需要同步更新case.json。

### 最终实验B：保存实际Viewer渲染结果

另开一轮接收，添加`--save-visuals`：

```bash
# GPU电脑，终端A
cd /home/zxn/disk/StreamDarkGS
bash tools/run_go2_receiver.sh --save-visuals
```

现场的隧道、Viewer和发送命令不变。输出增加：

```text
output/go2_live_XXXXXXXX/viewer_frames/
  albedo/       每次实际发布的Albedo渲染JPEG
  relit/        每次实际发布的重光照JPEG
  frames.jsonl  源图名、任务版本、文件名和时间元数据
```

保存复用已经编码的JPEG，不重复渲染，不是SAM标签图，也不是原始相机照片。
后端发布但浏览器未拉到的帧也会保存；被预览调度器提前合并、未渲染的帧不会保存。
文件名包含时间和版本，最终优化结果不会覆盖同视角早期预览。
文件中的保存时间不等于曝光时间或浏览器显示时间。

本轮同样输出时间CSV，但磁盘写入会影响渲染线程与队列，`viewer-save / save_ms`单独记录保存开销。
比较纯速度用实验A；观察图像质量和留档用实验B。

### 最终实验C：录制现场Viewer实际屏幕

无需安装远程录屏服务，也不需要把屏幕上传到GPU电脑。
在现场Ubuntu桌面浏览器打开`http://127.0.0.1:18765/profile?poll=50`：

1. 若是旧页面，刷新一次，确认出现“开始录屏 / 停止并保存”按钮。
2. **开始送图之前**点击“开始录屏”。浏览器会弹出屏幕共享选择框。
3. 要录整个Viewer页面，选择对应浏览器窗口或标签页；要录整个桌面，选择整个屏幕。
4. 确认页面提示“正在录制”，再启动现场发送脚本。
5. 保持Viewer在前台，等待输入、重建和最终优化结果显示完成。
6. 点击“停止并保存”，视频下载到现场Ubuntu浏览器的下载位置。
   若未自动下载，点击页面中的“下载录屏”链接。
7. 确认视频文件已下载并能播放后，再关闭页面或隧道。

默认不录声音，通常为`.webm`，浏览器支持情况不同也可能为`.mp4`。
浏览器录屏请求最高30 FPS，但不保证实际录制帧率；不会把重建刷新频率变成30 FPS。
录到的内容包括真实等待、画面更新和跳帧，区别于实验B的后端图片序列。

**隐私与可靠性：**

- 必须由你点击并选择共享范围，不能静默开始录屏。
- 选择整个屏幕会包含终端、通知、密码输入等其他窗口；建议关闭敏感内容，优先只录Viewer窗口。
- 当前录制片段暂存在浏览器内存，点击停止后才生成文件；页面崩溃/关闭可能丢失未保存录屏。
  首次按60秒短实验测试，长实验分段录制，不要连续录制数小时。
- GPU服务结束后页面网络请求可能报错，但本地录屏通常仍在继续，仍需手动“停止并保存”。
- 推荐桌面版Chrome/Chromium。需允许屏幕录制，Ubuntu/Wayland桌面可能还会弹出系统授权。
  若按钮报告不支持，确认经`localhost/127.0.0.1`隧道打开，而不是普通HTTP远程IP。
- 录屏会增加现场CPU/GPU负担；纯计时实验A不要录屏。可以在实验B同时录屏用于展示。
- 录屏功能已做代码检查，真实浏览器权限、桌面捕获和下载需要现场先用10秒试录验证。

现场Ubuntu的`~/go2_remote_demo/go2_captures/<时间>/`：

- `frame_000000.jpg`等：上传前保存的采样图片。
- `session.json`：本次配置。
- `events.jsonl`：保存、上传、确认、结束及错误记录。

GPU电脑启动时打印的`output/go2_live_XXXXXXXX/`：

- `input_frames/`：接收到的图片。
- `gaussian_map.pt`：完成后的高斯地图。
- `online_global_optimization_loss.json`：在线优化记录。
- `run.log`：全部模块计时和浏览器确认。

正常结束会自动生成`summary.csv`、`module_times.csv`。把CSV、完整`run.log`、优化历史
和输出目录路径发给我分析；展示材料可另外提供实验B图片或实验C录屏。

在GPU端检查（替换真实目录）：

```bash
rg '\[profile-run\]|\[save\]|\[viewer-client-timing\]' output/go2_live_XXXXXXXX/run.log
```

正常60张、create stride=2、每次优化10步，预期30次创建、30次在线优化、共300步。
预览提交数量可能为60次输入视角加一次最终更新；实际渲染/浏览器确认数量可因合并而更少。
优先用优化历史核对任务，而不是用预览数量推断优化是否完成。

`receive_to_ack_ms`包括GPU端收到图后至浏览器绘制确认返回的时间，是显示延迟近似上界。
不包含此前GO2拍摄、现场SDK取图和上传时间。跨电脑时间戳比较需校时；不要直接相减未同步的时钟。

## 11. 停止、断线与恢复

### 正常结束

让发送器完成指定张数，它会自动发送`/finish`。保持隧道和Viewer运行，等待GPU优化、保存结束。
最后才关闭隧道。`/finish`只结束输入，不会清空已接收任务。

### 想提前停止

1. 现场采集终端按一次Ctrl+C，停止继续取图。
2. 保持GPU程序和隧道运行。
3. 现场检查`/status`；已有图片且决定结束本轮时执行：

```bash
curl --max-time 10 -X POST http://127.0.0.1:18765/finish
```

4. 等GPU正常保存。若尚未收到任何图片，结束空会话可能报“没有帧”，重新开一轮即可。

### 上传超时、HTTP错误或SSH断开

发送脚本会停止、保留本地图片、记录错误，**不会自动重试上传，也不会自动发送finish**。
原因：当前服务器没有按客户端帧ID去重。请求超时可能是“后端已收到、确认没回来”，重传可能重复建图。

恢复时先重连隧道，查看`/status`、现场events和GPU日志；不要把同一批图盲目重发到原会话。
最简单的保守恢复是：让原会话用已收图片正常收尾；随后新开接收会话，从本地保存图片重新验证/重放。
这不是断点续建，也不会自动合并两张地图。若需要长时间无人值守和可靠续传，应另做带幂等ID的上传协议。

离线重放示例（新接收会话，图片数按实际修改）：

```bash
python send_go2_frames.py \
  --frames-dir go2_captures/实际采集目录 \
  --server http://127.0.0.1:18765 --fps 1 --count 10
```

## 12. 常见故障排查

| 现象 | 优先检查 |
|---|---|
| `No module named unitree_sdk2py` | 是否激活安装Python SDK的环境；是否只装了C++ SDK |
| SDK错误码、无法取图 | GO2网卡选择、机器人网络、相机服务、固件/SDK兼容性；先停在取图测试 |
| 连GO2后SSH断开 | 机器人网络是否抢占默认路由；恢复现场原有双网络配置 |
| 隧道端口占用 | `ss -ltnp \| grep 18765`；关闭自己旧的隧道，或改本地端口并同步修改URL |
| `administratively prohibited` | SSH服务是否允许TCP转发，请管理员检查 |
| `/status`连接拒绝 | 接收程序是否加载完、是否已退出、隧道是否运行、端口是否一致 |
| `Receiver is not fresh` | 上轮未重启或有另一个发送端；不要强行复用旧会话 |
| HTTP 429 | 输入队列满；停止上传、保存本地数据、检查后台吞吐，不无限重试 |
| 已收图但Viewer没画面 | 是否已收到完整窗口；GPU日志有无模型错误；页面是否为`/profile` |
| 有画面却无显示计时 | 页面保持前台；确认`viewer-client-timing`，不要只看`source_age_ms` |
| 画面成批跳动 | 窗口等待与最新优先预览所致；本版尚未加入相机位姿插值 |
| 床面/墙面错位或斑驳 | 先检查原图模糊、畸变、遮挡与Pi3位姿，不要先归咎于材质分割 |
| 结束后网页报连接失败 | 查看GPU是否已正常保存并退出；当前不是永久在线地图查看服务 |

端口和网络检查只查看状态，不建议随意关闭防火墙、杀未知进程或修改机器人网络。

## 13. 现场最短操作清单

1. 激活SDK环境，确认连接GO2的网卡，同时确认SSH仍可登录GPU电脑。
2. `--capture-only --count 3`取图并检查图片。
3. 终端A：GPU上启动`bash tools/run_go2_receiver.sh`。
4. 终端B：建立本地18765→GPU8765的SSH隧道。
5. 终端C：curl确认空会话；浏览器打开`/profile?poll=50`。
6. 先发送10张，确认画面、浏览器ACK和正常保存。
7. 新开接收会话，慢走采集60张。
8. 等待全部优化和保存，再关闭隧道；保留两端日志和图片。

## 官方参考

- [宇树Python SDK安装与使用](https://github.com/unitreerobotics/unitree_sdk2_python)
- [GO2前置相机示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/front_camera/camera_opencv.py)
- [宇树开发者文档](https://support.unitree.com/home/en/developer/)

不同SDK版本的示例目录可能不同，以现场安装版本为准。本手册的发送端使用
`ChannelFactoryInitialize(0, 网卡名)`及`VideoClient.GetImageSample()`，不调用运动控制API。
