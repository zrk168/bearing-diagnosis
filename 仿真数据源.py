"""
============================================================
轴向柱塞泵故障诊断仿真数据集生成器 v2
泵型: 10MCY14-1B  柱塞数: 7  公称压力: 31.5MPa
故障类型: 滑靴磨损 / 松靴 / 配流盘磨损 / 柱塞磨损 / 中心弹簧失效
故障程度: 轻度 / 中度 / 重度
工况: 10MPa / 20MPa / 31.5MPa
采样率: 10000 Hz  采样时长: 2s/样本

目录结构:
  pump_sim_dataset/
    train/
      normal_normal/
        train_normal_normal_10MPa_0000.csv
        train_normal_normal_20MPa_0001.csv
        train_normal_normal_31p5MPa_0002.csv
        ...  (300个文件，三种工况随机混合)
      slipper_wear_mild/
        ...
    val/
      ...
    test/
      ...
    metadata.csv
    label_fault_info.csv
    label_severity_info.csv

修复内容 vs v1:
  1. 去掉全局固定seed，每个样本独立seed保证多样性
  2. 每样本加入转速±2%、压力±3%随机扰动
  3. 泄漏量、故障幅值加入样本间随机性(±15%)
  4. Severity幅值差异压缩(0.25/0.55/1.00)，增加区分难度
  5. 去掉压力子文件夹，三种工况文件直接混合存放
============================================================
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 第一部分: 泵的基本物理参数
# ============================================================
class PumpParameters:
    z            = 7
    D_pitch      = 0.0710
    d_piston     = 0.0140
    beta_max     = np.radians(18)
    d_slipper    = 0.0220
    h_film_nom   = 12e-6
    r1_valve     = 0.0220
    r2_valve     = 0.0355
    h_valve_nom  = 8e-6
    Vg           = 14e-6
    P_nominal    = 31.5e6
    n_rated      = 1500
    eta_v_normal = 0.955
    eta_m_normal = 0.920
    rho          = 870.0
    mu           = 0.046
    beta_e       = 1.4e9
    f_shaft      = n_rated / 60
    f_piston     = f_shaft * z
    P_conditions = {'low': 10.0e6, 'medium': 20.0e6, 'high': 31.5e6}
    Q_theory     = Vg * n_rated / 60 * 1000 * 60
    Q_nominal    = Q_theory * eta_v_normal

PUMP = PumpParameters()


# ============================================================
# 第二部分: 传感器模型
# ============================================================
class SensorModel:
    def __init__(self, fs=10000):
        self.fs = fs

    def _quantize(self, value, v_min, v_max, bits=12):
        res = (v_max - v_min) / (2**bits - 1)
        return np.round((value - v_min) / res) * res + v_min

    def pressure_outlet(self, p_true):
        noise = np.random.normal(0, 0.002 * 40e6, len(p_true))
        p = np.clip(p_true + noise, 0, 40e6)
        return self._quantize(p, 0, 40e6, 12)

    def pressure_return(self, p_true):
        noise = np.random.normal(0, 0.010 * 1.0e6, len(p_true))
        return np.clip(p_true + noise, 0, 1.0e6)

    def flow_outlet(self, q_true):
        noise = np.random.normal(0, 0.005 * 20.0, len(q_true))
        return np.clip(q_true + noise, 2.0, 20.0)

    def flow_return(self, q_true):
        noise = np.random.normal(0, 0.010 * 10.0, len(q_true))
        return np.clip(q_true + noise, 0.5, 10.0)

    def vibration(self, acc_xyz):
        noise = np.random.normal(0, 0.001 * 9.81, acc_xyz.shape)
        acc = np.clip(acc_xyz + noise, -50*9.81, 50*9.81)
        b_hp, a_hp = butter(2, 0.5 / (self.fs/2), btype='high')
        b_lp, a_lp = butter(4, 4900 / (self.fs/2), btype='low')
        for i in range(3):
            acc[i] = filtfilt(b_hp, a_hp, acc[i])
            acc[i] = filtfilt(b_lp, a_lp, acc[i])
        return acc


# ============================================================
# 第三部分: 基础信号生成器
# ============================================================
class BaseSignalGenerator:
    def __init__(self, fs=10000):
        self.fs   = fs
        self.dt   = 1.0 / fs
        self.p    = PUMP
        self.sens = SensorModel(fs)

    def time_vector(self, duration=2.0):
        return np.arange(0, duration, self.dt)

    def _pressure_outlet_base(self, t, P_set, n_rpm=1500):
        omega = 2 * np.pi * n_rpm / 60
        p = np.ones_like(t) * P_set
        for k, A in enumerate([0.015, 0.005, 0.002, 0.001, 0.0005], 1):
            phi = np.random.uniform(0, 2*np.pi)
            p += P_set * A * np.sin(k * self.p.z * omega * t + phi)
        for k in range(1, 3):
            phi = np.random.uniform(0, 2*np.pi)
            p += P_set * 0.003/k * np.sin(k * omega * t + phi)
        p += P_set * 0.003 * np.sin(2*np.pi*2.0*t + np.random.uniform(0, 2*np.pi))
        p += np.random.normal(0, P_set * 0.001, len(t))
        return p

    def _pressure_return_base(self, t, P_set, n_rpm=1500):
        omega  = 2 * np.pi * n_rpm / 60
        P_back = 0.35e6 + P_set * 0.003
        p_ret  = np.ones_like(t) * P_back
        p_ret += P_back * 0.05 * np.sin(self.p.z * omega * t +
                                         np.random.uniform(0, 2*np.pi))
        p_ret += np.random.normal(0, P_back * 0.01, len(t))
        return np.clip(p_ret, 0.1e6, 0.8e6)

    def _flow_outlet_base(self, t, P_set, n_rpm=1500, eta_v=None):
        if eta_v is None:
            eta_v = self.p.eta_v_normal
        omega  = 2 * np.pi * n_rpm / 60
        Q_mean = self.p.Vg * (n_rpm / 60) * eta_v * 1000 * 60
        q = np.ones_like(t) * Q_mean
        for k, A in enumerate([0.012, 0.004, 0.002], 1):
            phi = np.random.uniform(0, 2*np.pi)
            q += Q_mean * A * np.sin(k * self.p.z * omega * t + phi + np.pi/2)
        q += np.random.normal(0, Q_mean * 0.002, len(t))
        return np.clip(q, 2.0, 22.0)

    def _flow_return_base(self, t, P_set, n_rpm=1500, leak_extra=0.0):
        Q_leak = self.p.Q_nominal * (1 - self.p.eta_v_normal) + 0.5 + leak_extra
        q_ret  = np.ones_like(t) * Q_leak
        q_ret += np.random.normal(0, Q_leak * 0.05, len(t))
        return np.clip(q_ret, 0.5, 8.0)

    def _vibration_base(self, t, P_set, n_rpm=1500):
        f_s    = n_rpm / 60
        f_p    = f_s * self.p.z
        A_base = 0.5 + P_set / self.p.P_nominal * 1.5
        acc    = np.zeros((3, len(t)))
        axis_scale = [1.0, 0.85, 0.70]
        for ax, sc in enumerate(axis_scale):
            for k in range(1, 6):
                phi = np.random.uniform(0, 2*np.pi)
                acc[ax] += sc * A_base * 0.25/k * 9.81 * np.sin(2*np.pi*k*f_s*t + phi)
            for k in range(1, 4):
                phi = np.random.uniform(0, 2*np.pi)
                acc[ax] += sc * A_base * 0.45/k * 9.81 * np.sin(2*np.pi*k*f_p*t + phi)
            acc[ax] += np.random.normal(0, 0.05 * A_base * 9.81, len(t))
        return acc

    def _package(self, t, p_out, p_ret, q_out, q_ret, acc,
                 fault_type, severity, P_set, n_rpm):
        p_out_m = self.sens.pressure_outlet(p_out)
        p_ret_m = self.sens.pressure_return(p_ret)
        q_out_m = self.sens.flow_outlet(q_out)
        q_ret_m = self.sens.flow_return(q_ret)
        acc_m   = self.sens.vibration(acc)

        f_s_actual = n_rpm / 60
        f_p_actual = f_s_actual * self.p.z

        df = pd.DataFrame({
            'time_s':              t,
            'pressure_outlet_MPa': p_out_m / 1e6,
            'pressure_return_MPa': p_ret_m / 1e6,
            'flow_outlet_Lmin':    q_out_m,
            'flow_return_Lmin':    q_ret_m,
            'acc_x_g':             acc_m[0] / 9.81,
            'acc_y_g':             acc_m[1] / 9.81,
            'acc_z_g':             acc_m[2] / 9.81,
            'fault_type':          fault_type,
            'severity':            severity,
            'load_pressure_MPa':   P_set / 1e6,
            'rpm':                 n_rpm,
            'shaft_freq_Hz':       f_s_actual,
            'piston_freq_Hz':      f_p_actual
        })
        return df


# ============================================================
# 第四部分: 各故障信号生成器
# ============================================================
# v2: Severity幅值压缩，轻度/中度更接近，增加分类难度
SEVERITY = {'mild': 0.25, 'moderate': 0.55, 'severe': 1.00}


# ------------------------------------------------------------------
# 正常工况
# ------------------------------------------------------------------
class NormalCondition(BaseSignalGenerator):
    def generate(self, severity='normal', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)
        t     = self.time_vector(duration)
        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        q_out = self._flow_outlet_base(t, P_set, n_rpm)
        q_ret = self._flow_return_base(t, P_set, n_rpm)
        acc   = self._vibration_base(t, P_set, n_rpm)
        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'normal', 'normal', P_set, n_rpm)


# ------------------------------------------------------------------
# 故障1: 滑靴磨损
# ------------------------------------------------------------------
class SlipperWear(BaseSignalGenerator):
    def generate(self, severity='mild', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)

        s   = SEVERITY[severity]
        # 故障幅值加入±15%样本间随机性
        s_r = s * np.random.uniform(0.85, 1.15)

        t   = self.time_vector(duration)
        f_s = n_rpm / 60
        f_p = f_s * self.p.z

        delta_eta = s_r * 0.09
        eta_v     = self.p.eta_v_normal - delta_eta

        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        for k in range(1, 5):
            A   = P_set * 0.010 * s_r / k**0.7
            phi = np.random.uniform(0, 2*np.pi)
            p_out += A * np.sin(2*np.pi * k * f_p * t + phi)
        f_hf  = 800 + s_r * 400
        p_out += P_set * 0.003 * s_r * np.sin(2*np.pi * f_hf * t +
                                                np.random.uniform(0, 2*np.pi))

        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        p_ret += 0.02e6 * s_r

        q_out = self._flow_outlet_base(t, P_set, n_rpm, eta_v=eta_v)

        leak_extra = s_r * np.random.uniform(1.5, 2.1)
        q_ret = self._flow_return_base(t, P_set, n_rpm, leak_extra=leak_extra)

        acc = self._vibration_base(t, P_set, n_rpm)
        for k in range(1, 4):
            A   = s_r * 0.8 / k * 9.81
            phi = np.random.uniform(0, 2*np.pi)
            for ax in range(3):
                sc = [1.0, 0.9, 0.7][ax]
                acc[ax] += sc * A * np.sin(2*np.pi * k * f_p * t + phi)

        hf_noise = np.random.normal(0, s_r * 0.6 * 9.81, len(t))
        b, a = butter(4, [500/(self.fs/2), 3000/(self.fs/2)], btype='band')
        hf_noise = filtfilt(b, a, hf_noise)
        acc[0] += 1.0 * hf_noise
        acc[1] += 0.9 * hf_noise
        acc[2] += 0.7 * hf_noise

        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'slipper_wear', severity, P_set, n_rpm)


# ------------------------------------------------------------------
# 故障2: 松靴
# ------------------------------------------------------------------
class LooseSlipper(BaseSignalGenerator):
    def generate(self, severity='mild', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)

        s   = SEVERITY[severity]
        s_r = s * np.random.uniform(0.85, 1.15)

        t   = self.time_vector(duration)
        f_s = n_rpm / 60
        f_p = f_s * self.p.z

        delta_eta = s_r * 0.07
        eta_v = self.p.eta_v_normal - delta_eta

        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        q_out = self._flow_outlet_base(t, P_set, n_rpm, eta_v=eta_v)
        q_ret = self._flow_return_base(t, P_set, n_rpm,
                                       leak_extra=s_r * np.random.uniform(1.9, 2.5))
        acc   = self._vibration_base(t, P_set, n_rpm)

        impulse_train = np.zeros_like(t)
        impact_period = 1.0 / f_p
        for ti in np.arange(0, duration, impact_period):
            jitter = np.random.normal(0, 0.00025)
            idx = int((ti + jitter) * self.fs)
            if 0 <= idx < len(t):
                impulse_train[idx] += 1.0 + 0.4 * np.random.randn()

        response_len = int(0.015 * self.fs)
        tt = np.arange(response_len) / self.fs
        f_res   = 1800 + 600 * s_r
        damping = 350 + 120 * s_r
        impulse_response = np.exp(-damping * tt) * np.sin(2*np.pi*f_res*tt)
        impact_signal = np.convolve(impulse_train, impulse_response, mode='same')

        p_out += P_set * 0.012 * s_r * impact_signal
        p_ret += 0.04e6  * s_r * impact_signal
        q_out += -0.25   * s_r * impact_signal
        q_ret +=  0.35   * s_r * np.abs(impact_signal)

        for ax in range(3):
            scale = [3.0, 2.4, 1.4][ax]
            acc[ax] += scale * s_r * 9.81 * impact_signal

        mod = 1 + 0.35 * s_r * np.sin(2*np.pi*f_s*t + np.random.uniform(0, 2*np.pi))
        for ax in range(3):
            acc[ax] *= mod

        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'loose_slipper', severity, P_set, n_rpm)


# ------------------------------------------------------------------
# 故障3: 配流盘磨损
# ------------------------------------------------------------------
class ValvePlateWear(BaseSignalGenerator):
    def generate(self, severity='mild', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)

        s   = SEVERITY[severity]
        s_r = s * np.random.uniform(0.85, 1.15)

        t   = self.time_vector(duration)
        f_s = n_rpm / 60
        f_p = f_s * self.p.z

        delta_eta = s_r * 0.11
        eta_v = self.p.eta_v_normal - delta_eta

        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        q_out = self._flow_outlet_base(t, P_set, n_rpm, eta_v=eta_v)

        pressure_factor = P_set / self.p.P_nominal
        leak_extra = s_r * np.random.uniform(0.9, 1.5) * (1.2 + 2.2 * pressure_factor)
        q_ret = self._flow_return_base(t, P_set, n_rpm, leak_extra=leak_extra)

        acc = self._vibration_base(t, P_set, n_rpm)

        switch_signal = np.zeros_like(t)
        period = 1.0 / f_p
        for ti in np.arange(0, duration, period):
            idx   = int(ti * self.fs)
            width = int((0.0008 + 0.0006*s_r) * self.fs)
            if idx + width < len(t):
                window = signal.windows.tukey(width, alpha=0.7)
                switch_signal[idx:idx+width] += window
        switch_signal -= np.mean(switch_signal)

        p_out += P_set * 0.018 * s_r * switch_signal
        for k in range(1, 6):
            A   = P_set * 0.008 * s_r / np.sqrt(k)
            phi = np.random.uniform(0, 2*np.pi)
            p_out += A * np.sin(2*np.pi*k*f_p*t + phi)

        p_ret += 0.08e6 * s_r + 0.03e6 * s_r * np.sin(2*np.pi*f_p*t)

        for k in range(1, 4):
            q_out += -0.18 * s_r / k * np.sin(2*np.pi*k*f_p*t +
                                                np.random.uniform(0, 2*np.pi))

        mf_noise = np.random.normal(0, s_r * 1.1 * 9.81, len(t))
        b, a = butter(4, [300/(self.fs/2), 1500/(self.fs/2)], btype='band')
        mf_noise = filtfilt(b, a, mf_noise)

        for ax in range(3):
            acc[ax] += [1.0, 0.9, 0.8][ax] * mf_noise
            for k in range(1, 5):
                acc[ax] += [1.2, 1.0, 0.8][ax] * s_r * 0.6/k * 9.81 * \
                           np.sin(2*np.pi*k*f_p*t + np.random.uniform(0, 2*np.pi))

        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'valve_plate_wear', severity, P_set, n_rpm)


# ------------------------------------------------------------------
# 故障4: 柱塞磨损
# ------------------------------------------------------------------
class PistonWear(BaseSignalGenerator):
    def generate(self, severity='mild', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)

        s   = SEVERITY[severity]
        s_r = s * np.random.uniform(0.85, 1.15)

        t   = self.time_vector(duration)
        f_s = n_rpm / 60
        f_p = f_s * self.p.z

        delta_eta = s_r * 0.10
        eta_v = self.p.eta_v_normal - delta_eta

        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        q_out = self._flow_outlet_base(t, P_set, n_rpm, eta_v=eta_v)

        pressure_factor = P_set / self.p.P_nominal
        leak_extra = s_r * np.random.uniform(0.8, 1.2) * (0.9 + 1.7 * pressure_factor)
        q_ret = self._flow_return_base(t, P_set, n_rpm, leak_extra=leak_extra)

        acc = self._vibration_base(t, P_set, n_rpm)

        modulation = 1 + 0.45 * s_r * np.sin(2*np.pi*f_s*t +
                                               np.random.uniform(0, 2*np.pi))
        for k in range(1, 5):
            A   = P_set * 0.009 * s_r / k
            phi = np.random.uniform(0, 2*np.pi)
            p_out += modulation * A * np.sin(2*np.pi*k*f_p*t + phi)

        q_out += -0.35 * s_r * np.sin(2*np.pi*f_s*t + np.random.uniform(0, 2*np.pi))
        q_ret +=  0.25 * s_r * np.sin(2*np.pi*f_s*t + np.random.uniform(0, 2*np.pi))

        broadband = np.random.normal(0, s_r * 0.9 * 9.81, len(t))
        b, a = butter(4, [200/(self.fs/2), 2500/(self.fs/2)], btype='band')
        broadband = filtfilt(b, a, broadband)

        for ax in range(3):
            scale = [1.1, 1.0, 0.75][ax]
            acc[ax] += scale * modulation * broadband
            for k in range(1, 4):
                acc[ax] += scale * s_r * 0.7/k * 9.81 * \
                           np.sin(2*np.pi*k*f_p*t + np.random.uniform(0, 2*np.pi))

        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'piston_wear', severity, P_set, n_rpm)


# ------------------------------------------------------------------
# 故障5: 中心弹簧失效
# ------------------------------------------------------------------
class CenterSpringFailure(BaseSignalGenerator):
    def generate(self, severity='mild', P_set=20e6, duration=2.0,
                 n_rpm=1500, sample_seed=None):
        if sample_seed is not None:
            np.random.seed(sample_seed)

        s   = SEVERITY[severity]
        s_r = s * np.random.uniform(0.85, 1.15)

        t   = self.time_vector(duration)
        f_s = n_rpm / 60
        f_p = f_s * self.p.z

        delta_eta = s_r * 0.045
        eta_v = self.p.eta_v_normal - delta_eta

        p_out = self._pressure_outlet_base(t, P_set, n_rpm)
        p_ret = self._pressure_return_base(t, P_set, n_rpm)
        q_out = self._flow_outlet_base(t, P_set, n_rpm, eta_v=eta_v)
        q_ret = self._flow_return_base(t, P_set, n_rpm,
                                       leak_extra=s_r * np.random.uniform(0.5, 0.7))
        acc   = self._vibration_base(t, P_set, n_rpm)

        f_low1   = 7.0 + 3.0 * np.random.rand()
        low_wave = (
            np.sin(2*np.pi*f_low1*t   + np.random.uniform(0, 2*np.pi)) +
            0.7  * np.sin(2*np.pi*f_s*t    + np.random.uniform(0, 2*np.pi)) +
            0.35 * np.sin(2*np.pi*2*f_s*t  + np.random.uniform(0, 2*np.pi))
        )

        p_out += P_set * 0.012 * s_r * low_wave
        p_ret += 0.035e6 * s_r * low_wave
        q_out += 0.45 * s_r * low_wave
        q_ret += 0.12 * s_r * low_wave

        acc[0] += 0.6 * s_r * 9.81 * low_wave
        acc[1] += 0.8 * s_r * 9.81 * low_wave
        acc[2] += 2.2 * s_r * 9.81 * low_wave

        impulse_train = np.zeros_like(t)
        for ti in np.arange(0, duration, 1/f_s):
            idx = int((ti + np.random.normal(0, 0.0006)) * self.fs)
            if 0 <= idx < len(t):
                impulse_train[idx] = 1.0

        response_len = int(0.025 * self.fs)
        tt       = np.arange(response_len) / self.fs
        response = np.exp(-180 * tt) * np.sin(2*np.pi*650*tt)
        impact   = np.convolve(impulse_train, response, mode='same')

        acc[2] += 1.5 * s_r * 9.81 * impact
        acc[1] += 0.8 * s_r * 9.81 * impact

        return self._package(t, p_out, p_ret, q_out, q_ret, acc,
                             'center_spring_failure', severity, P_set, n_rpm)


# ============================================================
# 第五部分: 数据集生成与划分
# ============================================================
class DatasetGenerator:
    def __init__(self, fs=10000, duration=2.0, output_dir='pump_sim_dataset'):
        self.fs         = fs
        self.duration   = duration
        self.output_dir = output_dir

        self.generators = {
            'normal':               NormalCondition(fs),
            'slipper_wear':         SlipperWear(fs),
            'loose_slipper':        LooseSlipper(fs),
            'valve_plate_wear':     ValvePlateWear(fs),
            'piston_wear':          PistonWear(fs),
            'center_spring_failure':CenterSpringFailure(fs)
        }

        self.pressures = {
            '10MPa':    10.0e6,
            '20MPa':    20.0e6,
            '31p5MPa':  31.5e6
        }

        self.train_per_class = 300
        self.val_per_class   = 100
        self.test_per_class  = 50

        self.class_list = [
            ('normal',               'normal'),
            ('slipper_wear',         'mild'),
            ('slipper_wear',         'moderate'),
            ('slipper_wear',         'severe'),
            ('loose_slipper',        'mild'),
            ('loose_slipper',        'moderate'),
            ('loose_slipper',        'severe'),
            ('valve_plate_wear',     'mild'),
            ('valve_plate_wear',     'moderate'),
            ('valve_plate_wear',     'severe'),
            ('piston_wear',          'mild'),
            ('piston_wear',          'moderate'),
            ('piston_wear',          'severe'),
            ('center_spring_failure','mild'),
            ('center_spring_failure','moderate'),
            ('center_spring_failure','severe'),
        ]

        self.fault_label_map = {
            'normal': 0, 'slipper_wear': 1, 'loose_slipper': 2,
            'valve_plate_wear': 3, 'piston_wear': 4, 'center_spring_failure': 5
        }
        self.severity_label_map = {
            'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3
        }

    def _allocate_pressures(self, total_count, base_seed=0):
        """均匀分配三种工况，返回随机打乱后的工况名列表"""
        pressure_names = list(self.pressures.keys())
        base  = total_count // 3
        rem   = total_count % 3
        counts = [base + (1 if i < rem else 0) for i in range(3)]

        sequence = []
        for name, cnt in zip(pressure_names, counts):
            sequence.extend([name] * cnt)

        rng = np.random.default_rng(base_seed)
        rng.shuffle(sequence)
        return sequence

    def _make_sample_path(self, split, fault_type, severity, pressure_name, idx):
        """v2: 去掉压力子文件夹，直接存在 fault_type_severity/ 下"""
        cls_dir = f'{fault_type}_{severity}'
        fname   = f'{split}_{fault_type}_{severity}_{pressure_name}_{idx:04d}.csv'
        return os.path.join(self.output_dir, split, cls_dir, fname)

    def _save_sample_csv(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')

    def generate_dataset(self):
        os.makedirs(self.output_dir, exist_ok=True)

        metadata         = []
        sample_global_id = 0

        split_config = {
            'train': self.train_per_class,
            'val':   self.val_per_class,
            'test':  self.test_per_class
        }

        for split_idx, (split, per_class_num) in enumerate(split_config.items()):
            for class_idx, (fault_type, severity) in enumerate(self.class_list):

                base_seed        = split_idx * 10000 + class_idx * 100
                pressure_sequence = self._allocate_pressures(per_class_num,
                                                              base_seed=base_seed)

                for local_idx, pressure_name in enumerate(pressure_sequence):
                    P_set_nominal = self.pressures[pressure_name]

                    # v2: 每个样本独立seed，压力/转速加入随机扰动
                    sample_seed = base_seed + local_idx
                    rng_meta    = np.random.default_rng(sample_seed)
                    P_set       = P_set_nominal * rng_meta.uniform(0.97, 1.03)
                    n_rpm       = int(1500 * rng_meta.uniform(0.98, 1.02))

                    df = self.generators[fault_type].generate(
                        severity    = severity,
                        P_set       = P_set,
                        duration    = self.duration,
                        n_rpm       = n_rpm,
                        sample_seed = sample_seed
                    )

                    save_path = self._make_sample_path(
                        split, fault_type, severity, pressure_name, local_idx
                    )
                    self._save_sample_csv(df, save_path)

                    metadata.append({
                        'sample_id':          sample_global_id,
                        'split':              split,
                        'fault_type':         fault_type,
                        'severity':           severity,
                        'class_name':         f'{fault_type}_{severity}',
                        'load_pressure_MPa':  P_set_nominal / 1e6,
                        'actual_pressure_MPa':P_set / 1e6,
                        'pressure_name':      pressure_name,
                        'rpm':                n_rpm,
                        'fs_Hz':              self.fs,
                        'duration_s':         self.duration,
                        'label_fault':        self.fault_label_map[fault_type],
                        'label_severity':     self.severity_label_map[severity],
                        'file_path':          save_path.replace('\\', '/')
                    })

                    sample_global_id += 1
                    print(f'[{sample_global_id:04d}] {save_path}')

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.output_dir, 'metadata.csv'),
                           index=False, encoding='utf-8-sig')

        pd.DataFrame([
            {'label_fault': 0, 'fault_type': 'normal',               '中文含义': '正常'},
            {'label_fault': 1, 'fault_type': 'slipper_wear',         '中文含义': '滑靴磨损'},
            {'label_fault': 2, 'fault_type': 'loose_slipper',        '中文含义': '松靴'},
            {'label_fault': 3, 'fault_type': 'valve_plate_wear',     '中文含义': '配流盘磨损'},
            {'label_fault': 4, 'fault_type': 'piston_wear',          '中文含义': '柱塞磨损'},
            {'label_fault': 5, 'fault_type': 'center_spring_failure','中文含义': '中心弹簧失效'},
        ]).to_csv(os.path.join(self.output_dir, 'label_fault_info.csv'),
                  index=False, encoding='utf-8-sig')

        pd.DataFrame([
            {'label_severity': 0, 'severity': 'normal',   '中文含义': '正常'},
            {'label_severity': 1, 'severity': 'mild',     '中文含义': '轻度'},
            {'label_severity': 2, 'severity': 'moderate', '中文含义': '中度'},
            {'label_severity': 3, 'severity': 'severe',   '中文含义': '重度'},
        ]).to_csv(os.path.join(self.output_dir, 'label_severity_info.csv'),
                  index=False, encoding='utf-8-sig')

        print('\n============================================================')
        print('数据集生成完成')
        print(f'输出目录  : {self.output_dir}')
        print(f'总样本数  : {len(metadata_df)}')
        print('训练集    : 4800  验证集: 1600  测试集: 800')
        print('16类，每类 train 300 / val 100 / test 50')
        print('三种工况混合存放，无压力子文件夹')
        print('============================================================')
        return metadata_df


# ============================================================
# 第六部分: 主程序
# ============================================================
if __name__ == '__main__':
    generator = DatasetGenerator(
        fs         = 10000,
        duration   = 2.0,
        output_dir = 'pump_sim_dataset'
    )
    metadata = generator.generate_dataset()
    print(metadata.head())
