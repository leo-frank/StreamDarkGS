#!/usr/bin/env python3
"""Serve one D435i color frame and aligned depth per request over a local TCP socket.

Run on the Go2 computer. The socket binds to loopback for SSH port forwarding.
Protocol: client sends b'F'; server replies with a 12-byte !III length header,
then UTF-8 JSON metadata, JPEG color, and 16-bit PNG aligned depth.
"""

import argparse
import json
import socket
import struct
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18800)
    parser.add_argument('--serial', default='317622073182')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--camera-fps', type=int, default=6)
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--require-resolution', action='store_true',
                        help='Fail instead of falling back to 640x480 color')
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--color-exposure-ms', type=float,
                        help='Use a fixed color exposure (disables color auto exposure)')
    parser.add_argument('--color-gain', type=float,
                        help='Fixed color gain; only valid with --color-exposure-ms')
    parser.add_argument('--color-white-balance-k', type=float,
                        help='Fixed color white balance in kelvin (disables auto white balance)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.jpeg_quality <= 100:
        parser.error('invalid port or JPEG quality')
    if args.color_exposure_ms is not None and args.color_exposure_ms <= 0:
        parser.error('--color-exposure-ms must be positive')
    if args.color_gain is not None and args.color_exposure_ms is None:
        parser.error('--color-gain requires --color-exposure-ms')
    if args.color_white_balance_k is not None and args.color_white_balance_k <= 0:
        parser.error('--color-white-balance-k must be positive')
    return args


def intrinsics_dict(value):
    return {
        'width': value.width, 'height': value.height,
        'fx': value.fx, 'fy': value.fy, 'cx': value.ppx, 'cy': value.ppy,
        'distortion_model': str(value.model), 'distortion_coefficients': list(value.coeffs),
    }


def color_frame_to_bgr(color, color_format, rs, cv2, np):
    if color_format == rs.format.yuyv:
        # pyrealsense2 exposes packed YUYV as one channel; OpenCV expects HxWx2.
        packed = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(
            color.get_height(), color.get_width(), 2
        )
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
    return np.asanyarray(color.get_data())


def color_sensor_for_profile(device, rs):
    for sensor in device.query_sensors():
        if any(stream.stream_type() == rs.stream.color for stream in sensor.get_stream_profiles()):
            return sensor
    raise RuntimeError('D435i color sensor not found')


def set_ranged_option(sensor, option, value, label):
    if not sensor.supports(option):
        raise RuntimeError(f'Color sensor does not support {label}')
    limits = sensor.get_option_range(option)
    if not limits.min <= value <= limits.max:
        raise ValueError(f'{label}={value:g} outside supported range '
                         f'[{limits.min:g}, {limits.max:g}] (step {limits.step:g})')
    sensor.set_option(option, value)
    return sensor.get_option(option)


def configure_color_exposure(sensor, rs, args):
    if args.color_exposure_ms is not None:
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        # D435i RGB exposure option uses 100-us units; frame metadata uses us.
        raw_exposure = round(args.color_exposure_ms * 10)
        actual = set_ranged_option(sensor, rs.option.exposure, raw_exposure,
                                   'color fixed exposure (100-us units)')
        print(f'[realsense] color fixed exposure={actual:g} x 100 us', flush=True)
        if args.color_gain is not None:
            gain = set_ranged_option(sensor, rs.option.gain, args.color_gain, 'color gain')
            print(f'[realsense] color fixed gain={gain:g}', flush=True)


def configure_color_white_balance(sensor, rs, args):
    if args.color_white_balance_k is None:
        return
    option = getattr(rs.option, 'white_balance', None)
    if option is None or not sensor.supports(option):
        raise RuntimeError('Color sensor does not support manual white balance')
    auto_option = getattr(rs.option, 'enable_auto_white_balance', None)
    if auto_option is not None and sensor.supports(auto_option):
        sensor.set_option(auto_option, 0)
    actual = set_ranged_option(sensor, option, args.color_white_balance_k,
                               'color white balance (kelvin)')
    print(f'[realsense] color fixed white balance={actual:g} K', flush=True)


def frame_metadata_or_none(frame, value):
    return frame.get_frame_metadata(value) if value is not None and frame.supports_frame_metadata(value) else None


def sensor_option_or_none(sensor, option):
    return sensor.get_option(option) if option is not None and sensor.supports(option) else None


def start_supported_profile(rs, args):
    """Try listed D435i modes, preferring RGB-D and the requested resolution."""
    color_sizes = [(args.width, args.height)]
    if not args.require_resolution and (args.width, args.height) != (640, 480):
        color_sizes.append((640, 480))
    attempts = []
    for with_depth in ([False] if args.no_depth else [True, False]):
        for width, height in color_sizes:
            depth_sizes = ((640, 360), (480, 270)) if height >= 720 else ((640, 480),)
            if not with_depth:
                depth_sizes = (None,)
            # YUYV uses less USB bandwidth than BGR8 for large color frames.
            color_formats = ((rs.format.yuyv, rs.format.bgr8) if width > 640
                             else (rs.format.bgr8, rs.format.yuyv))
            for depth_size in depth_sizes:
                for color_format in color_formats:
                    item = (width, height, color_format, depth_size)
                    if item not in attempts:
                        attempts.append(item)
    errors = []
    for width, height, color_format, depth_size in attempts:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, width, height, color_format, args.camera_fps)
        if depth_size is not None:
            config.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, args.camera_fps)
        label = f'color={width}x{height}@{args.camera_fps} {color_format} depth={depth_size}'
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            errors.append(f'{label}: {exc}')
            print(f'[realsense] unavailable {label}: {exc}', flush=True)
            continue
        print(f'[realsense] selected {label}', flush=True)
        return pipeline, profile, color_format, depth_size is not None
    raise RuntimeError('No D435i stream profile could start. Check camera ownership, '
                       'supported frame rates, and USB link.\n' + '\n'.join(errors))


def run(args):
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    pipeline, profile, color_format, has_depth = start_supported_profile(rs, args)
    device = profile.get_device()
    actual_serial = device.get_info(rs.camera_info.serial_number)
    if actual_serial != args.serial:
        pipeline.stop()
        raise RuntimeError(f'opened serial {actual_serial}, expected {args.serial}')
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_info = intrinsics_dict(color_profile.get_intrinsics())
    color_sensor = color_sensor_for_profile(device, rs)
    depth_scale = None if not has_depth else device.first_depth_sensor().get_depth_scale()
    align = None if not has_depth else rs.align(rs.stream.color)
    print(f'[realsense] serial={actual_serial} color={color_info} depth_scale={depth_scale}', flush=True)

    try:
        configure_color_exposure(color_sensor, rs, args)
        configure_color_white_balance(color_sensor, rs, args)
        if color_sensor.supports(rs.option.exposure):
            limits = color_sensor.get_option_range(rs.option.exposure)
            print(f'[realsense] color exposure option range={limits.min:g}..{limits.max:g} '
                  f'step={limits.step:g} (100-us units)', flush=True)
        if color_sensor.supports(rs.option.gain):
            limits = color_sensor.get_option_range(rs.option.gain)
            print(f'[realsense] color gain option range={limits.min:g}..{limits.max:g} '
                  f'step={limits.step:g}', flush=True)
        white_balance = getattr(rs.option, 'white_balance', None)
        if white_balance is not None and color_sensor.supports(white_balance):
            limits = color_sensor.get_option_range(white_balance)
            print(f'[realsense] color white balance range={limits.min:g}..{limits.max:g} '
                  f'step={limits.step:g} K', flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', args.port))
            listener.listen(1)
            print(f'[realsense] listening on 127.0.0.1:{args.port}', flush=True)
            while True:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(20)
                    print('[realsense] client connected', flush=True)
                    while connection.recv(1) == b'F':
                        frames = pipeline.wait_for_frames(10000)
                        if align is not None:
                            frames = align.process(frames)
                        color = frames.get_color_frame()
                        depth = None if not has_depth else frames.get_depth_frame()
                        if not color or (has_depth and not depth):
                            raise RuntimeError('incomplete color/depth frameset')
                        color_image = color_frame_to_bgr(color, color_format, rs, cv2, np)
                        ok, jpeg = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                        if not ok:
                            raise RuntimeError('color JPEG encoding failed')
                        depth_png = b''
                        if depth is not None:
                            depth_image = np.asanyarray(depth.get_data())
                            ok, encoded = cv2.imencode('.png', depth_image)
                            if not ok:
                                raise RuntimeError('aligned depth PNG encoding failed')
                            depth_png = encoded.tobytes()
                        exposure_raw = frame_metadata_or_none(
                            color, rs.frame_metadata_value.actual_exposure
                        )
                        metadata = {
                            'source': 'Intel RealSense D435i color', 'serial': actual_serial,
                            'color_intrinsics': color_info,
                            'color_format': str(color_format),
                            'color_frame_number': color.get_frame_number(),
                            'color_timestamp_ms': color.get_timestamp(),
                            'color_actual_exposure_raw': exposure_raw,
                            # D435i RGB returned 100 for a fixed 10-ms exposure;
                            # retain the raw value because SDK metadata units vary.
                            'color_actual_exposure_ms_estimate': (
                                exposure_raw / 10.0 if exposure_raw is not None else None
                            ),
                            'color_gain_level': frame_metadata_or_none(
                                color, rs.frame_metadata_value.gain_level),
                            'color_auto_exposure_frame': frame_metadata_or_none(
                                color, rs.frame_metadata_value.auto_exposure),
                            'color_exposure_option_100us': sensor_option_or_none(
                                color_sensor, rs.option.exposure),
                            'color_gain_option': sensor_option_or_none(color_sensor, rs.option.gain),
                            'color_auto_exposure_option': sensor_option_or_none(
                                color_sensor, rs.option.enable_auto_exposure),
                            'color_white_balance_kelvin': frame_metadata_or_none(
                                color, getattr(rs.frame_metadata_value, 'white_balance', None)),
                            'color_white_balance_option_kelvin': sensor_option_or_none(
                                color_sensor, getattr(rs.option, 'white_balance', None)),
                            'color_auto_white_balance_option': sensor_option_or_none(
                                color_sensor, getattr(rs.option, 'enable_auto_white_balance', None)),
                            'captured_at_unix_ns': time.time_ns(),
                            'depth_aligned_to_color': depth is not None,
                            'depth_scale_m_per_unit': depth_scale,
                            'depth_frame_number': depth.get_frame_number() if depth is not None else None,
                            'depth_timestamp_ms': depth.get_timestamp() if depth is not None else None,
                        }
                        info = json.dumps(metadata).encode('utf-8')
                        connection.sendall(struct.pack('!III', len(info), len(jpeg), len(depth_png)))
                        connection.sendall(info)
                        connection.sendall(jpeg)
                        if depth_png:
                            connection.sendall(depth_png)
                        print(f'[realsense] frame={metadata["color_frame_number"]} '
                              f'exposure_raw={exposure_raw} '
                              f'exposure_ms_est={metadata["color_actual_exposure_ms_estimate"]} '
                              f'gain={metadata["color_gain_level"]} '
                              f'jpeg={len(jpeg)} depth={len(depth_png)}', flush=True)
    finally:
        pipeline.stop()


if __name__ == '__main__':
    run(arguments())
