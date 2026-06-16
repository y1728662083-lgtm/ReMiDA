from __future__ import annotations

CONDITION_NAME_TO_ID = {
    "pronounced": 0,
    "inner": 1,
    "visualized": 2,
}
CONDITION_ID_TO_NAME = {v: k for k, v in CONDITION_NAME_TO_ID.items()}

CLASS_ID_TO_NAME = {
    0: "arriba",
    1: "abajo",
    2: "derecha",
    3: "izquierda",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}

DEFAULT_INNER_SPEECH_CHANNELS = [
    "Fp1","Fp2","AF3","AF4","F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "FC5","FC3","FC1","FCz","FC2","FC4","FC6",
    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
    "CP5","CP3","CP1","CPz","CP2","CP4","CP6",
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8",
]
AUX_CHANNEL_MARKERS = ("EOG", "EMG", "EXG", "AUX", "ECG")
