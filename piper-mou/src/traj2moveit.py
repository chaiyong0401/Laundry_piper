#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

ARM_FTJ_NS = '/arm_controller/follow_joint_trajectory'
JOINT_NAMES = ['joint1','joint2','joint3','joint4','joint5','joint6']
BOUNCE_JOINT = 1
AMP_DEG = 20.0
UP_TIME, DOWN_TIME, BACK_TIME = 1.0, 1.0, 1.0
ADD_VEL = True

BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

def wait_one_js(node: Node, timeout=5.0):
    got = {'msg': None}
    sub = node.create_subscription(
        JointState, '/joint_states', lambda m: got.__setitem__('msg', m), BEST_EFFORT_QOS
    )
    node.get_logger().info('waiting 1 message on /joint_states ...')
    end = node.get_clock().now().nanoseconds + int(timeout*1e9)
    while rclpy.ok() and got['msg'] is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.get_clock().now().nanoseconds > end:
            break
    node.destroy_subscription(sub)
    return got['msg']

class BounceToJTC(Node):
    def __init__(self):
        super().__init__('bounce_to_jtc_client')
        self.cli = ActionClient(self, FollowJointTrajectory, ARM_FTJ_NS)

        js = wait_one_js(self, timeout=5.0)
        if js is None:
            raise RuntimeError('joint_states timeout (QoS/네임스페이스/퍼블리셔 확인)')

        name_to_pos = {n: v for n, v in zip(js.name, js.position)}
        q0 = [float(name_to_pos[n]) for n in JOINT_NAMES]

        amp = math.radians(AMP_DEG)
        q_up = q0[:]; q_up[BOUNCE_JOINT] += amp
        q_dn = q0[:]; q_dn[BOUNCE_JOINT] -= amp

        jt = JointTrajectory(joint_names=JOINT_NAMES)
        def mkpt(pos, tfs, vel=None):
            p = JointTrajectoryPoint()
            p.positions = pos
            if vel is not None: p.velocities = vel
            p.time_from_start = Duration(sec=int(tfs), nanosec=int((tfs-int(tfs))*1e9))
            return p

        v_up = [0.0]*6; v_dn = [0.0]*6; v_bk = [0.0]*6
        if ADD_VEL:
            v_up[BOUNCE_JOINT] =  (amp)/max(UP_TIME,1e-6)
            v_dn[BOUNCE_JOINT] = (-2*amp)/max(DOWN_TIME,1e-6)
            v_bk[BOUNCE_JOINT] =  (amp)/max(BACK_TIME,1e-6)

        t=0.0
        jt.points += [
            mkpt(q0, t, [0.0]*6 if ADD_VEL else None),
            mkpt(q_up, t:=t+UP_TIME,  v_up if ADD_VEL else None),
            mkpt(q_dn, t:=t+DOWN_TIME, v_dn if ADD_VEL else None),
            mkpt(q0,  t:=t+BACK_TIME, v_bk if ADD_VEL else None),
        ]

        if not self.cli.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'no action server: {ARM_FTJ_NS}')

        goal = FollowJointTrajectory.Goal(trajectory=jt)
        self.get_logger().info(f'send JT → joints={jt.joint_names}, points={len(jt.points)}')
        self.cli.send_goal_async(goal).add_done_callback(self._goal_resp)

    def _goal_resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Goal REJECTED'); rclpy.shutdown(); return
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, fut):
        res = fut.result().result
        self.get_logger().info(f'JTC result: code={res.error_code} msg="{res.error_string}"')
        rclpy.shutdown()

def main():
    rclpy.init()
    try:
        node = BounceToJTC()
        rclpy.spin(node)
    except RuntimeError as e:
        print(f'[ERR] {e}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

