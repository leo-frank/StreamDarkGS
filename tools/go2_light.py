#!/usr/bin/env python3
"""Switch the GO2 VUI head light through the robot's ROS 2 request topic."""

import argparse
import json
import os
import sys
import time


SET_BRIGHTNESS = 1005
GET_BRIGHTNESS = 1006


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('on', 'off', 'status', 'toggle'))
    parser.add_argument('--level', type=int, default=8,
                        help='Brightness when turning on, 1..10 (default: 8)')
    parser.add_argument('--timeout', type=float, default=8,
                        help='Seconds to wait for each VUI response (default: 8)')
    args = parser.parse_args()
    if not 1 <= args.level <= 10 or args.timeout <= 0:
        parser.error('--level must be 1..10 and --timeout must be positive')
    return args


class VuiRosClient:
    def __init__(self, node, request_type, response_type, rclpy):
        self.node = node
        self.request_type = request_type
        self.rclpy = rclpy
        self.response = None
        self.pending_id = None
        self.publisher = node.create_publisher(request_type, '/api/vui/request', 10)
        self.subscriber = node.create_subscription(
            response_type, '/api/vui/response', self._on_response, 10)

    def _on_response(self, message):
        if message.header.identity.id == self.pending_id:
            self.response = message

    def call(self, api_id, parameter, timeout):
        request = self.request_type()
        self.pending_id = time.monotonic_ns() & ((1 << 63) - 1)
        self.response = None
        request.header.identity.id = self.pending_id
        request.header.identity.api_id = api_id
        request.parameter = json.dumps(parameter)
        deadline = time.monotonic() + timeout
        next_publish = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if self.publisher.get_subscription_count() and now >= next_publish:
                self.publisher.publish(request)
                next_publish = now + 0.5
            self.rclpy.spin_once(self.node, timeout_sec=min(0.1, max(0, deadline - now)))
            if self.response is not None:
                response = self.response
                if response.header.identity.api_id != api_id:
                    raise RuntimeError('GO2 VUI responded with a different API ID')
                code = response.header.status.code
                if code != 0:
                    raise RuntimeError(f'GO2 VUI returned code={code}: {response.data}')
                return response.data
        raise TimeoutError('No GO2 VUI response; check eth1 and /api/vui/request')

    def close(self):
        self.node.destroy_publisher(self.publisher)
        self.node.destroy_subscription(self.subscriber)


def brightness(client, timeout):
    result = json.loads(client.call(GET_BRIGHTNESS, {}, timeout))
    return int(result['brightness'])


def run(args):
    # Match the working onboard video source, even in a fresh SSH terminal.
    os.environ['ROS_DOMAIN_ID'] = '0'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
    os.environ['CYCLONEDDS_URI'] = (
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>eth1'
        '</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
    )
    try:
        import rclpy
        from unitree_api.msg import Request, Response
    except ImportError as exc:
        raise RuntimeError('Robot Python needs rclpy and unitree_api.msg') from exc

    rclpy.init()
    node = rclpy.create_node('go2_light_switch')
    client = VuiRosClient(node, Request, Response, rclpy)
    try:
        if args.action in ('status', 'toggle'):
            current = brightness(client, args.timeout)
            if args.action == 'status':
                print(f'brightness={current} ({"on" if current else "off"})')
                return
            target = 0 if current > 0 else args.level
        else:
            target = args.level if args.action == 'on' else 0

        client.call(SET_BRIGHTNESS, {'brightness': target}, args.timeout)
        actual = brightness(client, args.timeout)
        if actual != target:
            raise RuntimeError(f'GO2 reported brightness={actual}, expected {target}')
        print(f'brightness={actual} ({"on" if actual else "off"})')
    finally:
        client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        run(arguments())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
