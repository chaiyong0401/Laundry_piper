#!/usr/bin/env python3
"""Publish a one-shot ground-truth marker for transport event evaluation."""

from __future__ import annotations

import argparse
import datetime
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


ALIASES = {
    "p": "perturb_start",
    "l": "contact_loss",
    "e": "trial_end",
}


def wait_for_subscriber(node: Node, publisher, timeout: float) -> None:
    deadline = time.monotonic() + max(0.0, timeout)
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def publish_marker(
    node: Node,
    publisher,
    event: str,
    marker_file: Path,
    discovery_timeout: float = 2.0,
) -> None:
    timestamp_ns = time.monotonic_ns()
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    with marker_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "name": event,
            "timestamp_ns": timestamp_ns,
            "wall_time_iso": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }) + "\n")
        stream.flush()

    if publisher.get_subscription_count() == 0:
        wait_for_subscriber(node, publisher, discovery_timeout)
    if publisher.get_subscription_count() == 0:
        print(
            f"\r[EVENT-EVAL] locally recorded: {event}; "
            "ROS subscriber not found")
        return
    message = String()
    message.data = event
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.05)
    print(
        f"\r[EVENT-EVAL] published: {event:<16} "
        f"(subscribers={publisher.get_subscription_count()})")


def keyboard_mode(
    node: Node,
    publisher,
    marker_file: Path,
    discovery_timeout: float,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard mode requires an interactive terminal")
    print("[EVENT-EVAL] keyboard marker mode")
    print("  p: perturb_start   l: contact_loss   e: trial_end   q: quit")
    print("  Enter is not required.")
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            # Keep ROS discovery alive even when no key is pressed. This lets
            # the marker process be started before basket_ori.py.
            rclpy.spin_once(node, timeout_sec=0.0)
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            event = ALIASES.get(key)
            if event is not None:
                publish_marker(
                    node, publisher, event, marker_file, discovery_timeout)
    finally:
        termios.tcsetattr(
            sys.stdin, termios.TCSADRAIN, old_settings)


def validate_event(parser: argparse.ArgumentParser, event: str) -> str:
    event = ALIASES.get(event.lower(), event.lower())
    valid = {"perturb_start", "contact_loss", "trial_end"}
    if event.startswith("type="):
        trial_type = event.split("=", 1)[1]
        if trial_type not in {"stable", "abrupt", "gradual"}:
            parser.error("invalid trial type")
    elif event not in valid:
        parser.error(f"unknown event: {event}")
    return event


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "With no event argument, stay open and publish markers immediately "
            "when p/l/e is pressed."))
    parser.add_argument(
        "event",
        nargs="?",
        help=(
            "p/perturb_start, l/contact_loss, e/trial_end, "
            "or type=stable|abrupt|gradual; omit for keyboard mode"
        ),
    )
    parser.add_argument("--discovery_timeout", type=float, default=2.0)
    parser.add_argument(
        "--marker_file",
        default="/home/dyros/mcy_ws/transport_event_eval/marker_stream.jsonl",
        help="Append-only local marker stream used as a ROS-independent fallback",
    )
    args = parser.parse_args()
    event = (
        validate_event(parser, args.event)
        if args.event is not None else None)

    rclpy.init()
    node = Node("transport_event_marker")
    publisher = node.create_publisher(String, "/transport_eval/event", 10)
    wait_for_subscriber(node, publisher, args.discovery_timeout)
    marker_file = Path(args.marker_file).expanduser()
    try:
        if event is None:
            keyboard_mode(
                node, publisher, marker_file, args.discovery_timeout)
        else:
            publish_marker(
                node, publisher, event, marker_file,
                args.discovery_timeout)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
