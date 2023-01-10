import re
import numpy as np
from text.npu.symbols import *
import os

# Mappings from symbol to numeric ID and vice versa:
_ttsing_phone_to_id = {p: i for i, p in enumerate(ttsing_phone_set)}
_ttsing_pitch_to_id = {p: i for i, p in enumerate(ttsing_pitch_set)}
_ttsing_slur_to_id = {s: i for i, s in enumerate(ttsing_slur_set)}

ttsing_phone_to_int = {}
int_to_ttsing_phone = {}
for idx, item in enumerate(ttsing_phone_set):
    ttsing_phone_to_int[item] = idx
    int_to_ttsing_phone[idx] = item

ttsing_pitch_to_int = {}
int_to_ttsing_pitch = {}
for idx, item in enumerate(ttsing_pitch_set):
    ttsing_pitch_to_int[item] = idx
    int_to_ttsing_pitch[idx] = item

# opencpop
ttsing_opencpop_pitch_to_int = {}
for idx, item in enumerate(ttsing_opencpop_pitch_set):
    ttsing_opencpop_pitch_to_int[item] = idx

ttsing_slur_to_int = {}
int_to_ttsing_slur = {}
for idx, item in enumerate(ttsing_slur_set):
    ttsing_slur_to_int[item] = idx
    int_to_ttsing_slur[idx] = item


