"""Exercise SDL mapping/hotplug with an in-process virtual Xbox-style pad.

No OS input injection or physical controller is used. Optional pygame dependency.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


class GamepadDeviceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pygame"), "Optional pygame is not installed")
    def test_sdl_named_axes_buttons_and_disconnect(self):
        project = Path(__file__).resolve().parents[1]
        program = r'''
import ctypes
from pathlib import Path
import pygame as pg
from scripts.run_teleop import XboxDevice
pg.display.init()
pg.display.set_mode((32,32))
device = XboxDevice(pg)
libraries = list((Path(pg.__file__).resolve().parent.parent / 'pygame.libs').glob('libSDL2-2-*.so.*'))
if not libraries:
    raise SystemExit(77)
sdl = ctypes.CDLL(str(libraries[0]))
sdl.SDL_JoystickAttachVirtual.argtypes = [ctypes.c_int]*4
sdl.SDL_JoystickAttachVirtual.restype = ctypes.c_int
sdl.SDL_JoystickOpen.argtypes = [ctypes.c_int]
sdl.SDL_JoystickOpen.restype = ctypes.c_void_p
sdl.SDL_JoystickSetVirtualAxis.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int16]
sdl.SDL_JoystickSetVirtualButton.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_uint8]
sdl.SDL_JoystickClose.argtypes = [ctypes.c_void_p]
sdl.SDL_JoystickDetachVirtual.argtypes = [ctypes.c_int]
index = sdl.SDL_JoystickAttachVirtual(1,6,15,0)
assert index >= 0
device.index = index
handle = sdl.SDL_JoystickOpen(index)
assert handle
try:
    for axis in (4,5):
        assert sdl.SDL_JoystickSetVirtualAxis(handle,axis,-32768)==0
    pg.event.pump()
    pad = device.read()
    assert pad.connected and pad.neutral(), pad
    assert device.identity['mapping'] and device.identity['guid']
    for axis,value in enumerate((32767,-32768,16000,-16000,-32768,32767)):
        assert sdl.SDL_JoystickSetVirtualAxis(handle,axis,value)==0
    assert sdl.SDL_JoystickSetVirtualButton(handle,pg.CONTROLLER_BUTTON_RIGHTSHOULDER,1)==0
    assert sdl.SDL_JoystickSetVirtualButton(handle,pg.CONTROLLER_BUTTON_Y,1)==0
    pg.event.pump()
    pad = device.read()
    assert pad.lx==1 and pad.ly==-1 and .48<pad.rx<.50 and -.50<pad.ry<-.48, pad
    assert pad.lt==0 and pad.rt==1 and pad.down('rb') and pad.down('y'), pad
    sdl.SDL_JoystickClose(handle)
    handle = None
    assert sdl.SDL_JoystickDetachVirtual(index)==0
    pg.event.pump()
    assert not device.read().connected
finally:
    if handle:
        sdl.SDL_JoystickClose(handle)
        sdl.SDL_JoystickDetachVirtual(index)
    device.close()
    pg.quit()
print('SDL named axes, triggers, buttons and unplug passed')
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=project, capture_output=True, text=True,
                                env=dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS="1",
                                         PYGAME_HIDE_SUPPORT_PROMPT="1"))
        if result.returncode == 77:
            self.skipTest("Virtual-pad test requires the bundled Linux SDL2 wheel")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
