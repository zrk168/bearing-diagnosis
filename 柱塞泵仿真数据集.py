import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 2024
ROOT_DIR = r"E:\柱塞泵\仿真数据源2"

N_POINTS = 1000
FS = 1000
DURATION = N_POINTS / FS

PISTON_NUM = 7
DISPLACEMENT_ML_R = 10.0
WORKING_CONDITIONS = [10.0, 20.0, 31.5]

CLASSES_CONFIG = [
    {"folder": "normal",        "label": 0,  "fault_key": "normal",  "fault_name": "正常",       "sev_name": "normal",   "sev_id": -1},
    {"folder": "slipper_mild",  "label": 1,  "fault_key": "slipper", "fault_name": "滑靴磨损",   "sev_name": "轻度",     "sev_id": 1},
    {"folder": "slipper_medium","label": 2,  "fault_key": "slipper", "fault_name": "滑靴磨损",   "sev_name": "中度",     "sev_id": 2},
    {"folder": "slipper_severe","label": 3,  "fault_key": "slipper", "fault_name": "滑靴磨损",   "sev_name": "重度",     "sev_id": 3},
    {"folder": "loose_mild",    "label": 4,  "fault_key": "loose",   "fault_name": "松靴",       "sev_name": "轻度",     "sev_id": 1},
    {"folder": "loose_medium",  "label": 5,  "fault_key": "loose",   "fault_name": "松靴",       "sev_name": "中度",     "sev_id": 2},
    {"folder": "loose_severe",  "label": 6,  "fault_key": "loose",   "fault_name": "松靴",       "sev_name": "重度",     "sev_id": 3},
    {"folder": "valve_mild",    "label": 7,  "fault_key": "valve",   "fault_name": "配流盘磨损", "sev_name": "轻度",     "sev_id": 1},
    {"folder": "valve_medium",  "label": 8,  "fault_key": "valve",   "fault_name": "配流盘磨损", "sev_name": "中度",     "sev_id": 2},
    {"folder": "valve_severe",  "label": 9,  "fault_key": "valve",   "fault_name": "配流盘磨损", "sev_name": "重度",     "sev_id": 3},
    {"folder": "piston_mild",   "label": 10, "fault_key": "piston",  "fault_name": "柱塞磨损",   "sev_name": "轻度",     "sev_id": 1},
    {"folder": "piston_medium", "label": 11, "fault_key": "piston",  "fault_name": "柱塞磨损",   "sev_name": "中度",     "sev_id": 2},
    {"folder": "piston_severe", "label": 12, "fault_key": "piston",  "fault_name": "柱塞磨损",   "sev_name": "重度",     "sev_id": 3},
    {"folder": "spring_mild",   "label": 13, "fault_key": "spring",  "fault_name": "中心弹簧失效","sev_name": "轻度",    "sev_id": 1},
    {"folder": "spring_medium", "label": 14, "fault_key": "spring",  "fault_name": "中心弹簧失效","sev_name": "中度",    "sev_id": 2},
    {"folder": "spring_severe", "label": 15, "fault_key": "spring",  "fault_name": "中心弹簧失效","sev_name": "重度",    "sev_id": 3},
]

SPLIT_COUNTS = {"train": 300, "val": 100, "test": 50}


def generate_pump_signals(class_cfg, sample_seed):
    rng = np.random.RandomState(sample_seed)
    t = np.linspace(0, DURATION, N_POINTS, endpoint=False)

    rpm = rng.uniform(950.0, 1700.0)
    shaft_freq = rpm / 60.0
    piston_freq = shaft_freq * PISTON_NUM

    load_nominal = float(rng.choice(WORKING_CONDITIONS))
    load_actual = rng.uniform(load_nominal - 1.0, load_nominal + 1.0)

    q_theory = DISPLACEMENT_ML_R * rpm / 1000.0
    eta_v = rng.uniform(0.87, 0.94)
    base_leak = q_theory * (1.0 - eta_v)

    base_ripple = np.sin(2 * np.pi * piston_freq * t) + 0.3 * np.sin(4 * np.pi * piston_freq * t)

    pressure_in  = -0.25 + 0.05 * base_ripple + rng.normal(0, 0.03, N_POINTS)
    pressure_out = load_actual + 1.5 * base_ripple + rng.normal(0, 0.10, N_POINTS)
    flow_leak    = base_leak + rng.normal(0, 0.05, N_POINTS)
    flow_out     = (q_theory - base_leak) + 0.5 * base_ripple + rng.normal(0, 0.1, N_POINTS)
    vib_x = 0.05 * np.sin(2 * np.pi * shaft_freq * t)  + rng.normal(0, 0.010, N_POINTS)
    vib_y = 0.08 * np.sin(2 * np.pi * piston_freq * t + 1.5) + rng.normal(0, 0.015, N_POINTS)
    vib_z = 0.03 * base_ripple + rng.normal(0, 0.008, N_POINTS)

    fault_key = class_cfg["fault_key"]
    sf = max(class_cfg["sev_id"], 1)

    if fault_key == "slipper":
        coeff_vx   = rng.uniform(0.008, 0.025) * sf
        coeff_vy   = rng.uniform(0.012, 0.030) * sf
        fault_freq = rng.uniform(40.0, 60.0)
        p_coeff    = rng.uniform(0.25, 0.55) * sf
        leak_add   = rng.uniform(0.12, 0.28) * sf
        vib_x += rng.normal(0, coeff_vx, N_POINTS)
        vib_y += rng.normal(0, coeff_vy, N_POINTS)
        pressure_out -= p_coeff * np.sin(2 * np.pi * fault_freq * t)
        flow_leak += leak_add

    elif fault_key == "loose":
        impulse_amp = rng.uniform(0.25, 0.55) * sf
        decay_len   = rng.randint(20, 45)
        flow_coeff  = rng.uniform(0.20, 0.40) * sf
        impulses = np.zeros(N_POINTS)
        for k in range(int(DURATION * piston_freq)):
            idx = int(k / piston_freq * FS)
            if 0 <= idx < N_POINTS:
                impulses[idx] += impulse_amp
        kernel = np.exp(-np.linspace(0, 5, decay_len) ** 2)
        kernel /= kernel.sum()
        vib_z = np.convolve(vib_z + impulses, kernel, mode='same')[:N_POINTS]
        flow_out -= flow_coeff * np.abs(np.sin(2 * np.pi * piston_freq * t))

    elif fault_key == "valve":
        leak_add   = rng.uniform(0.5, 1.1) * sf
        flow_drop  = rng.uniform(0.4, 0.8) * sf
        p_coeff    = rng.uniform(1.0, 2.0) * sf
        phase      = rng.uniform(0, np.pi)
        vx_coeff   = rng.uniform(0.012, 0.030) * sf
        flow_leak += leak_add
        flow_out  -= flow_drop
        pressure_out += p_coeff * np.sin(2 * np.pi * piston_freq * t + phase)
        vib_x += vx_coeff * np.sin(2 * np.pi * 2 * piston_freq * t)

    elif fault_key == "piston":
        flow_drop  = rng.uniform(0.8, 1.6) * sf
        p_coeff    = rng.uniform(0.5, 1.1) * sf
        decay_rate = rng.uniform(3.0, 7.0)
        mod_ratio  = rng.uniform(0.3, 0.7)
        vy_coeff   = rng.uniform(0.018, 0.045) * sf
        flow_out  -= flow_drop
        pressure_out -= p_coeff * (1 - np.exp(-t * decay_rate))
        mod_f = shaft_freq * mod_ratio
        vib_y += vy_coeff * np.sin(2 * np.pi * mod_f * t) * np.sin(2 * np.pi * piston_freq * t)

    elif fault_key == "spring":
        low_ratio  = rng.uniform(0.2, 0.4)
        vz_coeff   = rng.uniform(0.025, 0.055) * sf
        flow_mod   = rng.uniform(0.07, 0.13) * sf
        p_coeff    = rng.uniform(0.7, 1.3) * sf
        low_f = shaft_freq * low_ratio
        vib_z += vz_coeff * np.sin(2 * np.pi * low_f * t)
        flow_out *= (1.0 - flow_mod * (0.5 + 0.5 * np.sin(2 * np.pi * low_f * t)))
        pressure_out += p_coeff * np.cos(2 * np.pi * low_f * t)

    flow_leak = np.maximum(flow_leak, 0.0)
    flow_out  = np.maximum(flow_out,  0.0)

    t0   = datetime(2024, 1, 1)
    czas = [(t0 + timedelta(seconds=float(tt))).strftime("%H:%M:%S.%f")[:-3] for tt in t]

    df = pd.DataFrame({
        "Czas2": czas, "Czas": czas,
        "Pressure - inlet":  pressure_in,
        "Pressure - outlet": pressure_out,
        "Flow - leakage":    flow_leak,
        "Flow - outlet":     flow_out,
        "Vibration X": vib_x,
        "Vibration Y": vib_y,
        "Vibration Z": vib_z,
        "stan": class_cfg["label"]
    })

    meta = {
        "split": "", "folder": class_cfg["folder"], "label": class_cfg["label"],
        "sample_name": "", "fault_key": class_cfg["fault_key"],
        "fault_name": class_cfg["fault_name"], "severity_name": class_cfg["sev_name"],
        "severity_id": class_cfg["sev_id"],
        "rpm": round(rpm, 3), "shaft_freq": round(shaft_freq, 5),
        "piston_freq": round(piston_freq, 5), "piston_num": PISTON_NUM,
        "load_nominal": load_nominal, "load_actual": round(load_actual, 5),
        "q_theory": round(q_theory, 5), "q_mean": round(float(np.mean(flow_out)), 5),
        "eta_v": round(eta_v, 6), "base_leak": round(base_leak, 5),
        "severity_str": f"{class_cfg['sev_name']}_{sf}x" if sf > 0 else "normal",
    }
    return df, meta


def main():
    print(f"输出目录: {ROOT_DIR}")
    for split in SPLIT_COUNTS:
        for cfg in CLASSES_CONFIG:
            os.makedirs(os.path.join(ROOT_DIR, split, cfg["folder"]), exist_ok=True)

    all_metadata = []
    global_sample_id = 0

    for split, count_per_class in SPLIT_COUNTS.items():
        print(f"生成 {split} 集...")
        for cfg in CLASSES_CONFIG:
            folder_path = os.path.join(ROOT_DIR, split, cfg["folder"])
            for i in range(count_per_class):
                sample_seed = SEED + global_sample_id
                global_sample_id += 1

                df, meta = generate_pump_signals(cfg, sample_seed)

                csv_name = f"{split}_{cfg['label']}_{i:04d}_{global_sample_id:06d}.csv"
                csv_path = os.path.join(folder_path, csv_name)
                df.to_csv(csv_path, index=False)

                meta["split"] = split
                meta["sample_name"] = csv_name
                meta["csv_path"] = csv_path
                all_metadata.append(meta)

    pd.DataFrame(all_metadata).to_csv(os.path.join(ROOT_DIR, "metadata.csv"), index=False)
    print(f"完成，共 {global_sample_id} 个样本")


if __name__ == "__main__":
    main()
