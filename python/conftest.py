# Makes `examples` importable (also in spawned worker processes, which inherit sys.path).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
