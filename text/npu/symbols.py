
ttsing_phone_set = ['_'] + [
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r",
    "s", "sh", "t", "x", "z", "zh", "a", "ai", "an", "ang", "ao", "e", "ei",
    "en", "eng", "er", "iii", "ii", "i", "ia", "ian", "iang", "iao", "ie", "in",
    "ing", "iong", "iou", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang",
    "uei", "uen", "ueng", "uo", "v", "van", "ve", "vn", "AH", "AA", "AO", "ER",
    "IH", "IY", "UH", "UW", "EH", "AE", "AY", "EY", "OY", "AW", "OW", "P", "B",
    "T", "D", "K", "G", "M", "N", "NG", "L", "S", "Z", "Y", "TH", "DH", "SH",
    "ZH", "CH", "JH", "V", "W", "F", "R", "HH", "AH0", "AA0", "AO0", "ER0",
    "IH0", "IY0", "UH0", "UW0", "EH0", "AE0", "AY0", "EY0", "OY0", "AW0", "OW0",
    "AH1", "AA1", "AO1", "ER1", "IH1", "IY1", "UH1", "UW1", "EH1", "AE1", "AY1",
    "EY1", "OY1", "AW1", "OW1", "AH2", "AA2", "AO2", "ER2", "IH2", "IY2", "UH2",
    "UW2", "EH2", "AE2", "AY2", "EY2", "OY2", "AW2", "OW2", "AH3", "AA3", "AO3",
    "ER3", "IH3", "IY3", "UH3", "UW3", "EH3", "AE3", "AY3", "EY3", "OY3", "AW3",
    "OW3", "D-1", "T-1", "P*", "B*", "T*", "D*", "K*", "G*", "M*", "N*", "NG*",
    "L*", "S*", "Z*", "Y*", "TH*", "DH*", "SH*", "ZH*", "CH*", "JH*", "V*",
    "W*", "F*", "R*", "HH*", "sp", "sil", "or", "ar", "aor", "our", "angr",
    "eir", "engr", "air", "ianr", "iaor", "ir", "ingr", "ur", "iiir", "uar",
    "uangr", "uenr", "iir", "ongr", "uor", "ueir", "iar", "iangr", "inr",
    "iour", "vr", "uanr", "ruai", "TR", "rest", 
    # opencpop
    'w', 'SP', 'AP', 'un', 'y', 'ui', 'iu'
]

ttsing_pitch_set = ['_'] + [
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C#/Db0", "C#/Db1", "C#/Db2",
    "C#/Db3", "C#/Db4", "C#/Db5", "C#/Db6", "D0", "D1", "D2", "D3", "D4", "D5",
    "D6", "D#/Eb0", "D#/Eb1", "D#/Eb2", "D#/Eb3", "D#/Eb4", "D#/Eb5", "D#/Eb6",
    "E0", "E1", "E2", "E3", "E4", "E5", "E6", "F0", "F1", "F2", "F3", "F4",
    "F5", "F6", "F#/Gb0", "F#/Gb1", "F#/Gb2", "F#/Gb3", "F#/Gb4", "F#/Gb5",
    "F#/Gb6", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G#/Ab0", "G#/Ab1",
    "G#/Ab2", "G#/Ab3", "G#/Ab4", "G#/Ab5", "G#/Ab6", "A0", "A1", "A2", "A3",
    "A4", "A5", "A6", "A#/Bb0", "A#/Bb1", "A#/Bb2", "A#/Bb3", "A#/Bb4",
    "A#/Bb5", "A#/Bb6", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "RestRest"
]

ttsing_opencpop_pitch_set = ['_'] + [
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", 
    "C#0/Db0", "C#1/Db1", "C#2/Db2", "C#3/Db3", "C#4/Db4", "C#5/Db5", "C#6/Db6", 
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", 
    "D#0/Eb0", "D#1/Eb1", "D#2/Eb2", "D#3/Eb3", "D#4/Eb4", "D#5/Eb5", "D#6/Eb6",
    "E0", "E1", "E2", "E3", "E4", "E5", "E6", 
    "F0", "F1", "F2", "F3", "F4", "F5", "F6", 
    "F#0/Gb0", "F#1/Gb1", "F#2/Gb2", "F#3/Gb3", "F#4/Gb4", "F#5/Gb5", "F#6/Gb6",
    "G0", "G1", "G2", "G3", "G4", "G5", "G6", 
    "G#0/Ab0", "G#1/Ab1", "G#2/Ab2", "G#3/Ab3", "G#4/Ab4", "G#5/Ab5", "G#6/Ab6", 
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", 
    "A#0/Bb0", "A#1/Bb1", "A#2/Bb2", "A#3/Bb3", "A#4/Bb4", "A#5/Bb5", "A#6/Bb6", 
    "B0", "B1", "B2", "B3", "B4", "B5", "B6", 
    "RestRest", "rest"
]

ttsing_slur_set = ['_'] + ['0', '1']


