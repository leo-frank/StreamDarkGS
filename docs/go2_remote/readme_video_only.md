# GO2 EDU：现场只录视频，回来再重建与测试

这份手册独立于实时上传方案。目标：**现场Ubuntu读取GO2自带相机并保存数据，
采集完成后把文件带回GPU电脑，再做重建、计时和可视化对照。**

现场不运行Pi3/MVInverse，不连接GPU服务，不需要SSH隧道或持续互联网连接。
录的是相机画面，不是Viewer录屏；脚本不控制机器狗运动，不录声音。

## 1. 出发前准备

需要：GO2 EDU、能连接GO2的Ubuntu电脑、已配置好的宇树Python SDK环境、足够磁盘空间。
用U盘或提前通过SSH复制这两个文件到现场电脑：

- `tools/record_go2_video.py`
- `docs/go2_remote/readme_video_only.md`

如果出发前可以从现场电脑SSH连接GPU电脑，下载示例（在现场Ubuntu运行）：

```bash
export GPU_SSH='用户名@GPU电脑地址'
mkdir -p ~/go2_video_capture
cd ~/go2_video_capture
scp "${GPU_SSH}:/home/zxn/disk/StreamDarkGS/tools/record_go2_video.py" .
scp "${GPU_SSH}:/home/zxn/disk/StreamDarkGS/docs/go2_remote/readme_video_only.md" .
```

非默认SSH端口请给scp加`-P 端口`，或使用已有SSH配置别名。
到现场后不需要继续与GPU电脑保持连接。

## 2. 现场先验证环境与网络

激活已安装宇树Python SDK的环境，检查：

```bash
python -c "import sys; print(sys.executable); import cv2, numpy; from unitree_sdk2py.go2.video.video_client import VideoClient; print('SDK OK')"
ip -br addr
ip route
df -h .
ffmpeg -version
```

与现场同事确认连接GO2的网卡，例如`enp3s0`，随后设置：

```bash
export GO2_IFACE='enp3s0'
```

上面的网卡名必须替换为实际值，不要选择Docker、VPN或不连接机器狗的网卡。
机器人IP、子网和SDK设置沿用现场已验证的配置，不要照抄网上的固定IP。
如果安装的只有C++ SDK，需要先配置Python SDK；不要运行运动控制示例来检查相机。

ffmpeg用于录制结束后生成MP4；没有ffmpeg也可以录制JPG与时间戳，回来再转视频，
不必在现场为了编码器重装环境。

## 3. 先录10秒试片

让机器狗静止，确认安全区域，运行：

```bash
cd ~/go2_video_capture
python record_go2_video.py \
  --interface "$GO2_IFACE" \
  --fps 15 \
  --seconds 10
```

终端打印`[output] ...`，数据保存在当前目录下：

```text
go2_videos/日期_时间/
  camera.mp4          录制结束后自动生成（需要ffmpeg及libx264）
  frames/             每次成功取得的图像，质量95 JPEG，保留原始尺寸
  timestamps.jsonl    每张图的取图完成时间、文件名、SDK请求耗时
  frames.ffconcat     按实际取图间隔生成视频的描述文件
  recording.json      实际帧数、目标/实测频率、尺寸、时长、完成状态
```

用现场文件管理器打开MP4，或者直接查看`frames/`中的图片。检查：

- 是GO2当前视角，画面没有冻结或连续重复旧帧。
- 方向、比例和亮度正常；不需要把横屏画面拉伸成手机竖屏。
- 没有严重模糊、黑屏或镜头遮挡。
- `recording.json`的`status`为`complete`，实际帧率足够。
- 若MP4没有生成，确认JPG和时间戳已保存；脚本的编码警告不等于取图数据丢失。

**15 FPS是取图目标，不保证相机实际输出15个新画面/秒。**
SDK调用、网络和JPEG保存慢时，实际频率会降低；脚本不会补造新的相机画面。
视频根据取图完成时间间隔生成，不是把所有图片强行按15 FPS播放。
时间是现场电脑收到SDK图像的时间，不是相机曝光时间；相机内部缓冲仍可能带来误差。

## 4. 正式采集：先录60秒，满意后再录较长片段

```bash
cd ~/go2_video_capture
python record_go2_video.py \
  --interface "$GO2_IFACE" \
  --fps 15 \
  --seconds 60
```

每次自动生成独立目录，不覆盖上一段。每段建议先控制在1～3分钟，便于发现问题、搬运和重复实验。
不要只按1 FPS录制，否则回来无法公平测试更密集的输入。

如果只想在现场采集，回来再编码MP4：

```bash
python record_go2_video.py \
  --interface "$GO2_IFACE" --fps 15 --seconds 60 --no-video
```

推荐拍摄顺序：

1. 起始位置停留约2～3秒，观察亮度与画面。
2. 遥控缓慢直行，保持相邻画面有充分重叠。
3. 缓慢转弯，避免急转、快跑或贴近墙面扫拍。
4. 回到此前经过的区域，拍摄回访视角，便于检查累计漂移。
5. 结束前短暂停留。

尽量固定光照，先选静态、纹理较丰富的场景。低视角遮挡和步态模糊都可能影响重建。
机身自带相机看不到的床面/桌面不能凭空重建；不要把覆盖不足误认为算法失败。
由人负责遥控与现场观察，不依赖尚未验证的重建结果决定机器狗动作。

## 5. 怎样停止？

达到`--seconds`后自动停止取图，再生成视频。
若要提前结束，按**一次Ctrl+C**，脚本会保留图片、写入时间戳清单并尝试生成视频。
看到完成信息后再关终端，不要在收尾期间再次中断或直接断电。

SDK错误会终止本段并写入`status: failed`；已保存的图像仍保留。
Ctrl+C结束为`interrupted`，可以查看已有片段，但不要当作完整时长实验。
单次SDK调用有超时，实际结束时刻可能比目标稍晚；以记录的时间为准。

## 6. 带回数据：复制整个采集目录

**推荐带回整个日期目录，不要只拷MP4。** JPG和时间戳有助于检查重复帧、SDK延迟，
也能避免后续只依赖有损视频重新抽帧。

可用U盘复制；或联网后在现场Ubuntu执行：

```bash
export GPU_SSH='用户名@GPU电脑地址'
ssh "$GPU_SSH" 'mkdir -p /home/zxn/disk/StreamDarkGS/data/go2_recordings'
scp -r go2_videos/实际日期目录 \
  "${GPU_SSH}:/home/zxn/disk/StreamDarkGS/data/go2_recordings/"
```

在GPU电脑确认文件数、`recording.json`和视频可播放后，再考虑清理现场副本；不要急着删除唯一原始数据。

## 7. GPU电脑：需要时从已录图片生成MP4

若现场没装ffmpeg或使用了`--no-video`，在GPU电脑执行（替换实际目录）：

```bash
cd /home/zxn/disk/StreamDarkGS/data/go2_recordings/实际日期目录
ffmpeg -hide_banner -n \
  -f concat -safe 0 -i frames.ffconcat \
  -vsync vfr \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
  -c:v libx264 -crf 18 -pix_fmt yuv420p camera.mp4
```

`-n`防止覆盖已有视频；需要重做时使用另一个输出文件名。
奇数宽高只在右/下边缘补齐至偶数，不拉伸内容。
MP4是便于查看和现有脚本处理的有损副本；时间会受ffmpeg时间基量化，末尾有用于保持时长的重复帧。
严格的帧时间分析应使用`timestamps.jsonl`和原JPG。

检查视频信息：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames:format=duration \
  -of default=noprint_wrappers=1 camera.mp4
```

不要只看MP4的标称帧率判断相机速度；实际取图频率看`recording.json`。

## 8. 回来先测试各模块耗时

下面在GPU电脑执行，替换视频路径，使用前60秒、1 FPS采样：

```bash
cd /home/zxn/disk/StreamDarkGS
bash run_pipeline_timing_matrix.sh \
  --mode matrix \
  --video '/home/zxn/disk/StreamDarkGS/data/go2_recordings/实际日期目录/camera.mp4' \
  --seconds 60 --fps 1 \
  --windows 3,5,10 \
  --create-strides 2 --repeats 3 \
  --online-optimization --online-steps 10 \
  --online-interval 1 --online-window-multiplier 5 \
  --low-latency-pipeline --preview-noncreation-frames \
  --timing wall
```

先保证视频足够长；若试片只有10秒，改成`--seconds 10`，不要假定短视频能产生60张图。
该命令重新抽帧，不要加旧实验的`--frames-dir`。想比较3 FPS，仅把`--fps 1`改为`--fps 3`，
保留同一视频和同一时间范围。视频本身实际采样很稀时，提高抽帧FPS可能只是重复图片，不能增加信息。

输出自动存到新的`output/pipeline_timing_.../`，包含`summary.csv`和`module_times.csv`。
这是无等待回放的计算耗时，不是现场拍摄到Viewer显示的真实远程延迟。
启用了非创建帧预览，与关闭该功能的旧计时也不能完全等价比较。

## 9. 回来观看重建、保存渲染、模拟实时输入

这一步仍不需要机器狗或现场电脑。使用当前GPU电脑本地的浏览器即可。

### 9.1 终端A：启动接收与渲染保存

```bash
cd /home/zxn/disk/StreamDarkGS
bash tools/run_go2_receiver.sh --save-visuals
```

该脚本固定W3/S2、create stride=2、在线优化10步、低延迟和非创建帧预览，
只监听本机8765端口。记录打印的输出目录。

### 9.2 浏览器打开Viewer

```text
http://127.0.0.1:8765/profile?poll=50
```

保持前台，需要展示录像时点击“开始录屏”，手动选择Viewer窗口。
这里录屏的是重建效果，不是本手册现场采集的GO2相机视频。

### 9.3 终端B：抽帧并按1 FPS送入

```bash
cd /home/zxn/disk/StreamDarkGS
GO2_REPLAY_DIR=$(mktemp -d "$PWD/output/go2_replay_frames_XXXXXXXX")
ffmpeg -hide_banner -loglevel warning \
  -i '/home/zxn/disk/StreamDarkGS/data/go2_recordings/实际日期目录/camera.mp4' \
  -t 60 -vf fps=1 -q:v 2 "$GO2_REPLAY_DIR/frame_%06d.jpg"
GO2_REPLAY_COUNT=$(find "$GO2_REPLAY_DIR" -maxdepth 1 -name '*.jpg' -type f | wc -l)
echo "本次发送 $GO2_REPLAY_COUNT 张"
/home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python tools/send_go2_frames.py \
  --frames-dir "$GO2_REPLAY_DIR" \
  --server http://127.0.0.1:8765 \
  --fps 1 --count "$GO2_REPLAY_COUNT"
```

文件夹模拟模式不导入宇树SDK。必须是空的新接收会话，发送完成会自动通知结束输入。
等待终端A完成全部优化、发布最终结果、保存高斯地图并生成CSV。
若开启浏览器录屏，点击“停止并保存”，下载完成后再关页面。

本轮输出：

- `gaussian_map.pt`：最终地图。
- `viewer_frames/albedo/`与`viewer_frames/relit/`：实际后端发布的渲染图。
- `viewer_frames/frames.jsonl`：源图和版本对应信息。
- `summary.csv`、`module_times.csv`、`run.log`：计时。
- 浏览器下载目录：手动录屏视频。

保存图片和录屏会增加开销，纯计时优先用第8节，效果留档用第9节。
这一步只模拟本机1 FPS到达；不包含GO2 SDK取图与远程网络传输延迟。
若想在别的电脑上看GPU端Viewer，可沿用实时手册的SSH隧道。

## 10. 如何判断录制与重建是否合格

首先检查数据：图像是否清晰、是否存在长时间冻结、转弯时是否仍有重叠，实际帧率是否够用。
再看重建：直线段是否稳定，转弯后墙面是否错位，回访区域是否出现重影和材质斑驳。
最后看时间：同一视频、同一时段、同一采样率比较不同窗口，不混入不同路线的差异。

建议保留“缓慢直行”“转弯”“返回起点”三个片段，先用W3测试，再用W5/W10对照。
不要仅因大窗口更准确就断定是帧数作用，它同时覆盖更长时间和更大视角变化。

## 11. 常见问题

| 问题 | 做法 |
|---|---|
| Python SDK导入失败 | 使用现场正确的SDK环境，确认不是只装了C++ SDK |
| `GetImageSample`错误 | 检查GO2网卡、网络与相机服务，先恢复静止取图 |
| 实际取图频率远低于15 FPS | 以实测为准；检查SDK/网络/磁盘，不能用重复帧伪装高FPS |
| 没有camera.mp4 | 查看ffmpeg/编码器警告；带回JPG、时间戳和ffconcat再转换 |
| 转视频速度较慢 | MP4转换在取图结束后进行，不影响已经记录的间隔，等待完成即可 |
| 视频尺寸异常 | 检查原JPG；脚本不旋转、不缩放，MP4仅补齐偶数边长 |
| 只带回了MP4 | 仍可按第8、9节测试，但无法完整检查原始取图时间和重编码影响 |
| 回放网页显示不了 | 接收服务是否启动、是否用了8765本机端口、是否已经收齐窗口 |

## 12. 官方接口参考与验证范围

[宇树官方GO2相机示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/front_camera/camera_opencv.py)
使用`VideoClient.GetImageSample()`获取图像；本脚本沿用此取图接口。

配套工具可做本地模拟录制/编码验证，但真实机器狗相机取流、实际采样频率、画面畸变及行走质量
仍需现场确认。本方案不请求或修改机器人运动状态。
