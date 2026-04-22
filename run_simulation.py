#!/usr/bin/env python3
from __future__ import annotations
"""
统一模拟入口。

按模式分发到各自独立的模拟器:
    - xray  -> simulate.py
    - laser -> simulate_laser.py

默认配置:
    - xray  -> config_simulate.py
    - laser -> config_simulate_laser.py

默认输出目录:
    - xray  -> result/xray/NNN
    - laser -> result/laser/NNN
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch to X-ray or laser simulation entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=("xray", "laser"),
        required=True,
        help="Simulation mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the mode-specific config file "
             "(xray: JSON override, laser: Python config file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple Gaussian spot mode (xray only)",
    )
    args = parser.parse_args()

    if args.mode == "laser" and args.simple:
        parser.error("--simple only applies to --mode xray")

    target = "simulate.py" if args.mode == "xray" else "simulate_laser.py"
    cmd = [sys.executable, target]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.simple:
        cmd.append("--simple")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
