#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import Optional

from IPython.display import clear_output
from IPython.display import Audio


TOOLS_DIR = Path(__file__).parent


def play_audio(filename: Path = TOOLS_DIR / 'bulb-horn-02.mp3') -> None:
    display(Audio(filename=str(filename), autoplay=True))


def run_timer(seconds: Optional[float] = None) -> None:
    if seconds is None:
        seconds = get_request()
        clear_output(wait=True)
    start = time.monotonic()
    try:
        end = start + seconds
    except:
        end = start
    while end > time.monotonic():
        print(f"Remaining: {end - time.monotonic():3.0f}"\
              f"\tElapsed: {time.monotonic() - start:3.0f}")
        time.sleep(1)
        clear_output(wait=True)
    play_audio()
    print("Time's Up!")


def get_request() -> float:
    try:
        resp = float(input("How Long (in minutes)? "))
    except:
        print("Invalid Input")
        return None
    if resp > 1000:
        print("Invalid Input")
        return None
    elif resp <= 0:
        print("Invalid Input")
        return None
    elif resp > 15:
        return resp
    else:
        return resp * 60
