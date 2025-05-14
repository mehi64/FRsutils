# frutil/__init__.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/models')))

import approximations, similarities, tnorms, implicators
from models import owafrs, vqrs, itfrs
