# Android 手机 HTTPS 实时采集配置

本文档说明如何配置 StreamDarkGS，使 Android 手机通过局域网 HTTPS 调用摄像头，并将图像实时上传到电脑进行重建和点亮预览。

## 1. 工作方式

数据链路如下：

```text
Android Chrome 摄像头
        ↓ HTTPS / Wi-Fi
电脑 CameraStreamServer
        ↓
Pi3 + MVInverse 滑动窗口
        ↓
Gaussian 增量融合
        ↓
网页 Albedo 与点亮预览
```

手机和电脑只需要连接同一个局域网。证书配置完成后，采集期间不需要连接 USB。

## 2. 前置条件

- 电脑和 Android 手机连接同一个 Wi-Fi。
- 路由器没有开启客户端隔离（AP Isolation）。
- 电脑已安装 OpenSSL。
- 项目位于：

```text
/home/zxn/disk/StreamDarkGS
```

- 项目使用的 HTTPS 端口默认为 `8765`。

## 3. 查询电脑局域网 IP

在电脑执行：

```bash
hostname -I
```

本次配置使用的电脑 IP 为：

```text
192.168.0.100
```

后续证书中的 IP 和手机访问的 IP 必须完全一致。如果电脑 IP 发生变化，需要按照第 10 节重新签发服务器证书。

## 4. 创建本地 CA

进入项目并创建证书目录：

```bash
cd /home/zxn/disk/StreamDarkGS
mkdir -p certs
cd certs
```

生成 CA 私钥：

```bash
openssl genrsa -out streamdarkgs-ca.key 4096
```

生成 CA 证书：

```bash
openssl req -x509 -new -nodes \
  -key streamdarkgs-ca.key \
  -sha256 \
  -days 3650 \
  -out streamdarkgs-ca.crt \
  -subj "/CN=StreamDarkGS Local CA"
```

安全注意事项：

- `streamdarkgs-ca.key` 是 CA 私钥，只能保存在电脑端。
- 不要把 CA 私钥发送到手机。
- 不要把 CA 私钥或服务器私钥提交到 Git。
- 手机只需要安装 `streamdarkgs-ca.crt`。

## 5. 生成服务器证书

### 5.1 生成服务器私钥

```bash
openssl genrsa -out streamdarkgs-server.key 2048
```

### 5.2 创建证书请求配置

创建 `certs/server.cnf`：

```ini
[req]
prompt = no
distinguished_name = dn
req_extensions = req_ext

[dn]
CN = 192.168.0.100

[req_ext]
subjectAltName = @alt_names

[alt_names]
IP.1 = 192.168.0.100
DNS.1 = localhost
```

### 5.3 创建服务器证书扩展配置

创建 `certs/server-ext.cnf`：

```ini
authorityKeyIdentifier = keyid,issuer
basicConstraints = CA:FALSE
keyUsage = digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
IP.1 = 192.168.0.100
DNS.1 = localhost
```

### 5.4 生成证书签名请求

```bash
openssl req -new \
  -key streamdarkgs-server.key \
  -out streamdarkgs-server.csr \
  -config server.cnf
```

### 5.5 使用本地 CA 签发服务器证书

```bash
openssl x509 -req \
  -in streamdarkgs-server.csr \
  -CA streamdarkgs-ca.crt \
  -CAkey streamdarkgs-ca.key \
  -CAcreateserial \
  -out streamdarkgs-server.crt \
  -days 825 \
  -sha256 \
  -extfile server-ext.cnf
```

### 5.6 设置私钥权限

```bash
chmod 600 streamdarkgs-ca.key streamdarkgs-server.key
chmod 644 streamdarkgs-ca.crt streamdarkgs-server.crt \
  streamdarkgs-server.csr server.cnf server-ext.cnf
```

### 5.7 验证证书

验证签名链：

```bash
openssl verify \
  -CAfile streamdarkgs-ca.crt \
  streamdarkgs-server.crt
```

正常结果：

```text
streamdarkgs-server.crt: OK
```

检查证书中的 IP：

```bash
openssl x509 \
  -in streamdarkgs-server.crt \
  -noout -subject -issuer -dates -ext subjectAltName
```

输出中必须包含：

```text
IP Address:192.168.0.100
DNS:localhost
```

## 6. 配置 StreamDarkGS 启动脚本

实时采集使用独立脚本：

```text
run_our11_textcluster_clean_stream.sh
```

脚本需要使用 stream 输入模式：

```bash
--input_mode stream
```

HTTPS 相关参数为：

```bash
--stream_host "0.0.0.0" \
--stream_port "8765" \
--stream_certfile "certs/streamdarkgs-server.crt" \
--stream_keyfile "certs/streamdarkgs-server.key" \
```

默认手机采集参数为：

```bash
--stream_capture_fps "2" \
--stream_jpeg_quality "85" \
--stream_queue_size "120" \
```

其余 Pi3、MVInverse、窗口大小、窗口步长、融合和优化参数与原有流程共享。

建议每次实验使用不同的 `TEST_RUN_DIR`，避免覆盖之前的 Gaussian map 和调试结果。

## 7. Android 安装本地 CA

只把以下文件传到手机：

```text
certs/streamdarkgs-ca.crt
```

不要传输以下文件：

```text
streamdarkgs-ca.key
streamdarkgs-server.key
```

Android 的安装入口通常为：

```text
设置
→ 安全
→ 更多安全设置
→ 加密与凭据
→ 安装证书
→ CA 证书
```

不同厂商的菜单名称可能略有差异。选择 `streamdarkgs-ca.crt`，确认安装为 CA 证书。部分 Android 系统会要求先设置锁屏密码。

安装完成后，完全退出并重新打开 Chrome。

## 8. 启动和采集

### 8.1 启动电脑端服务

```bash
cd /home/zxn/disk/StreamDarkGS
./run_our11_textcluster_clean_stream.sh
```

程序会先加载 Pi3、MVInverse 和 SAM2。等待终端显示类似信息：

```text
[stream] phone capture page: https://<本机局域网IP>:8765/
[stream] waiting for frames
```

### 8.2 手机打开采集页面

在 Android Chrome 中访问：

```text
https://192.168.0.100:8765/
```

必须满足：

- 使用 `https://`，不能使用 `http://`。
- IP 必须与服务器证书中的 SAN IP 一致。
- 手机与电脑位于同一个局域网。

### 8.3 开始采集

1. 点击“开始采集”。
2. Chrome 请求摄像头权限时选择允许。
3. 缓慢移动手机，保持相邻画面有足够重叠。
4. 页面会显示已发送帧数。
5. 累计到一个窗口后，后端开始 Pi3、MVInverse 和 Gaussian 融合。
6. Viewer 会显示最新 Albedo 和点亮结果。
7. 调整光源 X、Y 和环境光滑块；新的融合结果会使用当前灯光参数。

### 8.4 结束采集

拍摄完成后点击“结束并重建”，不要直接终止电脑程序。

后端随后会：

1. 停止接收新帧；
2. 处理最后不足一个窗口的帧；
3. 保存相机轨迹；
4. 完成配置中的优化；
5. 保存 Gaussian map 和 PLY；
6. 导出最终点亮结果。

等待电脑端程序自然结束。

## 9. 输出数据

每次 stream 运行会在 `TEST_RUN_DIR` 中创建独立采集目录：

```text
stream_capture_YYYYMMDD_HHMMSS/
```

典型输出结构：

```text
TEST_RUN_DIR/
├── stream_capture_YYYYMMDD_HHMMSS/  # 手机上传的原始 JPEG
├── gaussian_map.pt                  # Gaussian 完整状态
├── gaussian_map.ply                 # PLY 导出
├── cameras.json                     # 相机内参与轨迹
├── mature_material_debug/           # 材质调试结果
├── global_opt_render_debug/         # 优化调试结果
└── relit_flash_full/                # 最终点亮结果
```

## 10. 电脑 IP 变化后重新签发证书

假设新 IP 为 `192.168.0.120`：

1. 将 `server.cnf` 中的两处旧 IP 改为新 IP。
2. 将 `server-ext.cnf` 中的旧 IP 改为新 IP。
3. 重新生成签名请求：

```bash
cd /home/zxn/disk/StreamDarkGS/certs
openssl req -new \
  -key streamdarkgs-server.key \
  -out streamdarkgs-server.csr \
  -config server.cnf
```

4. 重新签发服务器证书：

```bash
openssl x509 -req \
  -in streamdarkgs-server.csr \
  -CA streamdarkgs-ca.crt \
  -CAkey streamdarkgs-ca.key \
  -CAcreateserial \
  -out streamdarkgs-server.crt \
  -days 825 \
  -sha256 \
  -extfile server-ext.cnf
```

5. 重新启动 StreamDarkGS。
6. 手机访问新 IP，例如 `https://192.168.0.120:8765/`。

只要继续使用原来的 CA，手机通常不需要重新安装 CA 证书。

为了避免 IP 经常变化，建议在路由器中为电脑配置 DHCP 静态租约。

## 11. 常见问题

### 11.1 `Cannot read properties of undefined (reading 'getUserMedia')`

原因通常是使用了非安全 HTTP 页面：

```text
http://192.168.0.100:8765/
```

改为：

```text
https://192.168.0.100:8765/
```

并确认 Android 已安装并信任 `streamdarkgs-ca.crt`。

### 11.2 浏览器提示证书名称或地址不匹配

手机访问的 IP 与服务器证书 SAN 中的 IP 不一致。查询电脑当前 IP，并按照第 10 节重新签发服务器证书。

### 11.3 手机打不开页面

依次检查：

- 电脑端是否已经显示 `[stream] waiting for frames`；
- 手机和电脑是否连接同一个 Wi-Fi；
- 手机能否访问电脑 IP；
- 防火墙是否允许 TCP 端口 `8765`；
- 路由器是否开启了客户端隔离；
- 地址是否使用正确的 HTTPS IP 和端口。

### 11.4 页面打开但摄像头没有权限

进入：

```text
Android 设置 → 应用 → Chrome → 权限 → 相机
```

设置为“仅在使用应用时允许”，然后完全关闭并重新打开 Chrome。

### 11.5 长时间没有 Viewer 结果

当前配置需要累计足够帧数形成第一个窗口。例如 `window_size=10`、采集速度为 2 FPS 时，至少需要约 5 秒采集时间，随后还需要等待模型推理和融合。

观察电脑终端是否依次出现：

```text
[stream] frame=... accepted=...
[window] index=...
[mvinverse] window=...
[fusion] ... created=...
```

### 11.6 输入队列已满

如果后端处理速度明显低于上传速度，页面可能收到 HTTP 429。可以降低脚本中的：

```bash
STREAM_CAPTURE_FPS=1
```

也可以适当增大 `STREAM_QUEUE_SIZE`，但这会增加内存占用和处理延迟。

## 12. 安全清单

- 仅将 `streamdarkgs-ca.crt` 安装到手机。
- 不发送或提交任何 `.key` 文件。
- 不向不受信任设备安装本地 CA。
- Demo 完成后，如果不再使用，可以从 Android 的用户凭据中删除该 CA。
- 只在受信任的局域网中监听 `0.0.0.0:8765`。
- 证书和服务器绑定 IP 发生变化时及时重新签发。
