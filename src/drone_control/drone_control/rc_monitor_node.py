"""
RC Monitor Node - ROS2 + MAVSDK
Đọc flight mode trực tiếp từ Pixhawk
Khi gạt switch RC → GUIDED/OFFBOARD → publish /auto_mode = True

Test SITL  : CONNECTION_STRING = "udpin://127.0.0.1:14550"
Orange Pi  : CONNECTION_STRING = "serial:///dev/serial/by-id/usb-Holybro_Pixhawk6C_290035001751333337363133-if00:115200"
"""

import asyncio
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from mavsdk import System


# ── Đổi dòng này khi cần ──────────────────────
CONNECTION_STRING = "serial:///dev/serial/by-id/usb-Holybro_Pixhawk6C_290035001751333337363133-if00:115200"
# CONNECTION_STRING = "udpin://127.0.0.1:14550"  # SITL


class RCMonitorNode(Node):

    def __init__(self):
        super().__init__('rc_monitor_node')
        self.get_logger().info("RC Monitor Node started (MAVSDK)...")

        self.drone     = System()
        self.loop      = asyncio.new_event_loop()
        self.last_mode = ""

        # Publishers
        self.auto_mode_pub = self.create_publisher(Bool,   '/auto_mode',   10)
        self.mode_pub      = self.create_publisher(String, '/flight_mode', 10)

        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.create_task(self._monitor_flight_mode())
        self.loop.run_forever()

    async def _connect(self):
        self.get_logger().info(f"Connecting: {CONNECTION_STRING}")
        await self.drone.connect(system_address=CONNECTION_STRING)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("Connected!")
                break

    async def _monitor_flight_mode(self):
        async for flight_mode in self.drone.telemetry.flight_mode():
            mode = str(flight_mode)

            if mode != self.last_mode:
                self.get_logger().info(f"Flight mode: {mode}")

                if self.last_mode in ["GUIDED", "OFFBOARD"] and mode not in ["GUIDED", "OFFBOARD"]:
                    self.get_logger().info(f"AUTO MODE OFF → {mode}")

                self.last_mode = mode

            # Publish flight mode
            mode_msg = String()
            mode_msg.data = mode
            self.mode_pub.publish(mode_msg)

            # Publish auto mode
            auto_msg = Bool()
            auto_msg.data = mode in ["GUIDED", "OFFBOARD"]
            self.auto_mode_pub.publish(auto_msg)

            if mode in ["GUIDED", "OFFBOARD"]:
                self.get_logger().info("AUTO MODE ON!")


def main(args=None):
    rclpy.init(args=args)
    node = RCMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()