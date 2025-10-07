BRICS_CLASSES = ["no_tumor", "glioma", "meningioma", "pituitary"]

BRICS_SHORT_CODE_MAPPING = {c[:2]: c for c in BRICS_CLASSES}
BRICS_CLASS_IDX_MAPPING = {c: i for i, c in enumerate(BRICS_CLASSES)}
