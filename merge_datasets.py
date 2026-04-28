"""
Merge dataset1, dataset2, dataset5 → merged_dataset
- Filters to 8 target labels only
- Remaps DS1 "No Stopping" → "no stopping or standing"
- Fixes DS2 train-val annotation leakage (removes leaked valid/test images)
- Re-indexes all class IDs to canonical 0-7
- Fresh 80/10/10 random re-split
"""

import os, glob, shutil, random, collections
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/home/nnmax/Desktop/forge-alpha/signboard-yolo/extracted")
OUT       = Path("/home/nnmax/Desktop/forge-alpha/signboard-yolo/merged_dataset")
SEED      = 42
SPLIT     = (0.80, 0.10, 0.10)   # train / valid / test

# ── Canonical class list (final 0-7) ──────────────────────────────────────────
CANONICAL = [
    "Cross Road",            # 0
    "Men At Work",           # 1
    "No Entry",              # 2
    "No Parking",            # 3
    "No Stopping or Standing",  # 4
    "Pedestrian Crossing",   # 5
    "Pedestrian Prohibited", # 6
    "School Ahead",          # 7
]

# ── Per-dataset original class lists ──────────────────────────────────────────
DS1_CLASSES = ['All Motor Vehicle Prohibited', 'Barrier Ahead', 'Bullock And Handcart Prohibited', 'Compulsary Ahead', 'Compulsary Keep Left', 'Compulsary Keep Right', 'Compulsary Sound Horn', 'Compulsary Turn Left', 'Compulsary Turn Right', 'Cross Road', 'Cycle Prohibited', 'Give Way', 'Horse Car Prohibited', 'Left Hand Curve', 'Left Reverse Bend', 'Left Turn Prohibited', 'Major Road Ahead', 'Men At Work', 'Narrow Bridge', 'Narrow Road Ahead', 'Narrow Road Left Ahead', 'No Entry', 'No Heavy Vehicles', 'No Parking', 'No Stopping', 'Pedestrian Crossing', 'Pedestrian Prohibited', 'Right Hair Pin Bend', 'Right Hand Curve', 'Right Reverse Bend', 'Right Turn', 'Right Turn Prohibited', 'Road Wideness Ahead', 'Roundabout', 'School Ahead', 'Side Road', 'Slippery Road', 'Sound Horn', 'Speed Limit', 'Steep Ascent', 'Stop', 'Straight Prohibited', 'T Junction', 'Traffic  Signal', 'Two Way Signs', 'U Turn Prohibited', 'Width Limit', 'Y Intersection']
DS2_CLASSES = ['ALL_MOTOR_VEHICLE_PROHIBITED', 'AXLE_LOAD_LIMIT', 'BARRIER_AHEAD', 'BULLOCK_AND_HANDCART_PROHIBITED', 'BULLOCK_PROHIBITED', 'CATTLE', 'COMPULSARY_AHEAD', 'COMPULSARY_AHEAD_OR_TURN_LEFT', 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 'COMPULSARY_CYCLE_TRACK', 'COMPULSARY_KEEP_LEFT', 'COMPULSARY_KEEP_RIGHT', 'COMPULSARY_MINIMUM_SPEED', 'COMPULSARY_SOUND_HORN', 'COMPULSARY_TURN_LEFT', 'COMPULSARY_TURN_LEFT_AHEAD', 'COMPULSARY_TURN_RIGHT', 'COMPULSARY_TURN_RIGHT_AHEAD', 'CROSS_ROAD', 'CYCLE_CROSSING', 'CYCLE_PROHIBITED', 'DANGEROUS_DIP', 'DIRECTION', 'FALLING_ROCKS', 'FERRY', 'GAP_IN_MEDIAN', 'GIVE_WAY', 'GUARDED_LEVEL_CROSSING', 'HANDCART_PROHIBITED', 'HEIGHT_LIMIT', 'HORN_PROHIBITED', 'HUMP_OR_ROUGH_ROAD', 'LEFT_HAIR_PIN_BEND', 'LEFT_HAND_CURVE', 'LEFT_REVERSE_BEND', 'LEFT_TURN_PROHIBITED', 'LENGTH_LIMIT', 'LOAD_LIMIT', 'LOOSE_GRAVEL', 'MEN_AT_WORK', 'NARROW_BRIDGE', 'NARROW_ROAD_AHEAD', 'NO_ENTRY', 'NO_PARKING', 'NO_STOPPING_OR_STANDING', 'OVERTAKING_PROHIBITED', 'PASS_EITHER_SIDE', 'PEDESTRIAN_CROSSING', 'PEDESTRIAN_PROHIBITED', 'PRIORITY_FOR_ONCOMING_VEHICLES', 'QUAY_SIDE_OR_RIVER_BANK', 'RESTRICTION_ENDS', 'RIGHT_HAIR_PIN_BEND', 'RIGHT_HAND_CURVE', 'RIGHT_REVERSE_BEND', 'RIGHT_TURN_PROHIBITED', 'ROAD_WIDENS_AHEAD', 'ROUNDABOUT', 'SCHOOL_AHEAD', 'SIDE_ROAD_LEFT', 'SIDE_ROAD_RIGHT', 'SLIPPERY_ROAD', 'SPEED_LIMIT_15', 'SPEED_LIMIT_20', 'SPEED_LIMIT_30', 'SPEED_LIMIT_40', 'SPEED_LIMIT_5', 'SPEED_LIMIT_50', 'SPEED_LIMIT_60', 'SPEED_LIMIT_70', 'SPEED_LIMIT_80', 'STAGGERED_INTERSECTION', 'STEEP_ASCENT', 'STEEP_DESCENT', 'STOP', 'STRAIGHT_PROHIBITED', 'TONGA_PROHIBITED', 'TRAFFIC_SIGNAL', 'TRUCK_PROHIBITED', 'TURN_RIGHT', 'T_INTERSECTION', 'UNGUARDED_LEVEL_CROSSING', 'U_TURN_PROHIBITED', 'WIDTH_LIMIT', 'Y_INTERSECTION']
DS5_CLASSES = ['all motor vehicle prohibited', 'axle load limit', 'bullock cart and hand cart prohibited', 'cattle ahead', 'chevron direction', 'compulsary ahead', 'compulsary ahead or turn left', 'compulsary ahead or turn right', 'compulsary keep left', 'compulsary keep right', 'compulsary sound horn', 'compulsary turn left ahead', 'compulsary turn right ahead', 'cross road', 'cycle crossing', 'cycle prohibited', 'dangerous dip', 'falling rocks', 'gap in median', 'give way', 'guarded level crossing', 'height limit', 'horn prohibited', 'hospital ahead', 'hump or rough road', 'left hand curve', 'left reverse bend', 'left turn prohibited', 'length limit', 'loose gravel', 'men at work', 'narrow bridge ahead', 'narrow road ahead', 'no entry', 'no parking', 'no stopping or standing', 'overtaking prohibited', 'pass either side', 'pedestrian crossing', 'pedestrian prohibited', 'petrol pump ahead', 'quay side or river bank', 'restriction ends', 'right hand curve', 'right reverse bend', 'right turn prohibited', 'road widens ahead', 'roundabout', 'school ahead', 'side road left', 'side road right', 'slippery road', 'speed limit 100', 'speed limit 120', 'speed limit 15', 'speed limit 20', 'speed limit 30', 'speed limit 40', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'staggered intersection', 'steep ascent', 'steep descent', 'stop', 'straight prohibited', 't intersection', 'traffic signal', 'truck prohibited', 'u turn', 'u turn prohibited', 'unguarded level crossing', 'width limit', 'y intersection']

# ── Helpers ───────────────────────────────────────────────────────────────────
def norm(s):
    return s.lower().replace('_', ' ').strip()

# Canonical label → 0-7 index
CANONICAL_NORM_TO_IDX = {norm(c): i for i, c in enumerate(CANONICAL)}
# Also add "no stopping" as alias
CANONICAL_NORM_TO_IDX["no stopping"] = 4  # DS1 fix

def build_remap(class_list):
    """original_idx -> canonical_idx (or None if not a target)"""
    remap = {}
    for orig_idx, cls_name in enumerate(class_list):
        n = norm(cls_name)
        if n in CANONICAL_NORM_TO_IDX:
            remap[orig_idx] = CANONICAL_NORM_TO_IDX[n]
    return remap

DS1_REMAP = build_remap(DS1_CLASSES)
DS2_REMAP = build_remap(DS2_CLASSES)
DS5_REMAP = build_remap(DS5_CLASSES)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def list_images(img_dir):
    return [
        f for f in glob.glob(str(img_dir / '*'))
        if Path(f).suffix.lower() in IMG_EXTS
    ]

def make_annotation_sig(label_path, remap):
    """
    Returns frozenset of 'canonical_idx|cx cy w h' strings for target lines,
    or None if file missing / no target lines.
    """
    if not label_path.exists():
        return None
    lines = label_path.read_text().strip().splitlines()
    sig_parts = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        orig_idx = int(parts[0])
        if orig_idx in remap:
            coords = ' '.join(f"{float(x):.4f}" for x in parts[1:])
            sig_parts.append(f"{remap[orig_idx]}|{coords}")
    if not sig_parts:
        return None
    return frozenset(sig_parts)

def rewrite_label(label_path, remap):
    """Returns list of rewritten annotation lines (only target classes, re-indexed)."""
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        orig_idx = int(parts[0])
        if orig_idx in remap:
            new_idx = remap[orig_idx]
            out.append(f"{new_idx} {' '.join(parts[1:])}")
    return out

# ── Step 1: Fix DS2 leakage — collect train annotation sigs ───────────────────
print("=" * 60)
print("Step 1: Collecting DS2 train annotation signatures for leakage fix...")
ds2_train_sigs = set()
ds2_train_img_dir = BASE / "dataset2" / "train" / "images"
ds2_train_lbl_dir = BASE / "dataset2" / "train" / "labels"

for img_path in list_images(ds2_train_img_dir):
    stem = Path(img_path).stem
    lbl_path = ds2_train_lbl_dir / (stem + ".txt")
    sig = make_annotation_sig(lbl_path, DS2_REMAP)
    if sig is not None:
        ds2_train_sigs.add(sig)

print(f"  DS2 train unique target annotation signatures: {len(ds2_train_sigs)}")

# ── Step 2: Collect all qualifying images into pool ───────────────────────────
print("\nStep 2: Collecting all qualifying images into pool...")

# pool entry: (src_img_path, label_lines, pool_filename)
pool = []
stats = collections.Counter()
leaked_skipped = 0

DATASETS = [
    ("dataset1", DS1_REMAP, "ds1"),
    ("dataset2", DS2_REMAP, "ds2"),
    ("dataset5", DS5_REMAP, "ds5"),
]

for ds_name, remap, prefix in DATASETS:
    ds_path = BASE / ds_name
    for split in ("train", "valid", "test"):
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        for img_path in list_images(img_dir):
            img_path = Path(img_path)
            stem = img_path.stem
            lbl_path = lbl_dir / (stem + ".txt")

            # Get annotation sig (target only)
            sig = make_annotation_sig(lbl_path, remap)
            if sig is None:
                continue   # no target labels in this image

            # DS2 leakage fix: skip valid/test images whose sig is in DS2 train
            if ds_name == "dataset2" and split in ("valid", "test"):
                if sig in ds2_train_sigs:
                    leaked_skipped += 1
                    continue

            # Rewrite label lines with canonical indices
            label_lines = rewrite_label(lbl_path, remap)
            if not label_lines:
                continue

            # Build pool filename (prefix to avoid collisions)
            new_stem = f"{prefix}_{stem}"
            new_filename = new_stem + img_path.suffix.lower()

            pool.append((img_path, label_lines, new_filename))

            # Track label counts
            for line in label_lines:
                cls_idx = int(line.split()[0])
                stats[CANONICAL[cls_idx]] += 1

print(f"  Total images in pool: {len(pool)}")
print(f"  DS2 leaked images removed (valid+test): {leaked_skipped}")
print(f"  Per-label counts in pool:")
for label in CANONICAL:
    print(f"    {label}: {stats[label]}")

# ── Step 3: Fresh 80/10/10 random split ───────────────────────────────────────
print("\nStep 3: Shuffling and splitting 80/10/10...")
random.seed(SEED)
random.shuffle(pool)

n = len(pool)
n_train = int(n * SPLIT[0])
n_valid = int(n * SPLIT[1])
n_test  = n - n_train - n_valid

train_pool = pool[:n_train]
valid_pool = pool[n_train:n_train + n_valid]
test_pool  = pool[n_train + n_valid:]

print(f"  Train: {len(train_pool)} | Valid: {len(valid_pool)} | Test: {len(test_pool)}")

# ── Step 4: Write merged dataset ──────────────────────────────────────────────
print("\nStep 4: Writing merged dataset to disk...")

# Clean output dir if exists
if OUT.exists():
    shutil.rmtree(OUT)

split_dirs = {}
for split_name in ("train", "valid", "test"):
    img_out = OUT / split_name / "images"
    lbl_out = OUT / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    split_dirs[split_name] = (img_out, lbl_out)

def write_split(split_data, split_name):
    img_out, lbl_out = split_dirs[split_name]
    written = 0
    per_label = collections.Counter()
    for src_img, label_lines, new_filename in split_data:
        # Copy image
        dst_img = img_out / new_filename
        shutil.copy2(src_img, dst_img)
        # Write label
        stem = Path(new_filename).stem
        dst_lbl = lbl_out / (stem + ".txt")
        dst_lbl.write_text("\n".join(label_lines) + "\n")
        written += 1
        for line in label_lines:
            per_label[int(line.split()[0])] += 1
    return written, per_label

tr_n, tr_lbl = write_split(train_pool, "train")
va_n, va_lbl = write_split(valid_pool, "valid")
te_n, te_lbl = write_split(test_pool,  "test")

print(f"  Written — train: {tr_n}, valid: {va_n}, test: {te_n}")

# ── Step 5: Write data.yaml ────────────────────────────────────────────────────
yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CANONICAL)}
names: {CANONICAL}
"""
(OUT / "data.yaml").write_text(yaml_content)
print("\ndata.yaml written.")

# ── Step 6: Final report ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)
print(f"Output directory: {OUT}")
print(f"\nSplit counts: train={tr_n}, valid={va_n}, test={te_n}, total={tr_n+va_n+te_n}")
print(f"\nDS2 leaked images removed: {leaked_skipped}")

print("\nPer-label annotation counts in TRAIN:")
for i, label in enumerate(CANONICAL):
    print(f"  [{i}] {label}: {tr_lbl[i]}")

print("\nPer-label annotation counts in VALID:")
for i, label in enumerate(CANONICAL):
    print(f"  [{i}] {label}: {va_lbl[i]}")

print("\nPer-label annotation counts in TEST:")
for i, label in enumerate(CANONICAL):
    print(f"  [{i}] {label}: {te_lbl[i]}")

print("\nAll annotation counts (train+valid+test):")
total_lbl = collections.Counter()
for d in [tr_lbl, va_lbl, te_lbl]:
    for k, v in d.items():
        total_lbl[k] += v
for i, label in enumerate(CANONICAL):
    print(f"  [{i}] {label}: {total_lbl[i]}")
print(f"  GRAND TOTAL: {sum(total_lbl.values())}")
print("\nDone!")
