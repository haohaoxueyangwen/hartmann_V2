from __future__ import annotations

import copy
from typing import Dict, Any, Optional, Tuple


def build_simulated_wfe(window_size_m: float,
                        settings: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[float]]:
    """
    Build (wfe_type, wfe_params, wfe_rms_rad) tuple based on the provided settings.

    Parameters
    ----------
    window_size_m : float
        Simulation window size in meters (used to derive pupil radius when missing).
    settings : Dict[str, Any]
        Dictionary containing:
            enable: bool
            mode: 'simple' | 'advanced'
            simple_params: dict
            advanced_params: dict
            tilt_tx_radpm: float
            tilt_ty_radpm: float
            zernike_scale: float
            psd_rms_scale: float

    Returns
    -------
    (wfe_type, wfe_params, wfe_rms) : tuple
        Values ready to pass into simulate_pulse_superposition. All None if disabled.
    """
    if not settings.get('enable', False):
        return None, None, None

    mode = str(settings.get('mode', 'advanced')).lower()

    if mode == 'simple':
        params = copy.deepcopy(settings.get('simple_params', {}))
        params['tilt_tx'] = float(params.get('tilt_tx', 0.0)) + float(settings.get('tilt_tx_radpm', 0.0))
        params['tilt_ty'] = float(params.get('tilt_ty', 0.0)) + float(settings.get('tilt_ty_radpm', 0.0))
        return 'tilt+defocus+astig', params, None

    if mode == 'advanced':
        params = copy.deepcopy(settings.get('advanced_params', {}))
        pupil = float(params.get('pupil_radius_m') or 0.0)
        if pupil <= 0.0:
            params['pupil_radius_m'] = float(window_size_m) * 0.5
        scale = float(settings.get('zernike_scale', 1.0))
        zernike_list = []
        for idx, coef in params.get('zernike_coefs', []):
            zernike_list.append((int(idx), float(coef) * scale))
        params['zernike_coefs'] = zernike_list
        if params.get('psd_rms_rad') is not None:
            params['psd_rms_rad'] = float(params['psd_rms_rad']) * float(settings.get('psd_rms_scale', 1.0))
        params['psd_corr_len_m'] = float(params.get('psd_corr_len_m', 0.0))
        params['tilt_tx'] = float(settings.get('tilt_tx_radpm', 0.0))
        params['tilt_ty'] = float(settings.get('tilt_ty_radpm', 0.0))
        psd_enabled = bool(params.get('psd_rms_rad')) and params['psd_rms_rad'] > 0.0
        wfe_type = 'zernike+psd' if psd_enabled else 'zernike'
        wfe_type = wfe_type + '+tilt'
        wfe_rms = params['psd_rms_rad'] if psd_enabled else None
        return wfe_type, params, wfe_rms

    if mode == 'geometric':
        params = copy.deepcopy(settings.get('geometric_params', {}))
        # Add global tilt if needed
        params['tilt_tx'] = float(settings.get('tilt_tx_radpm', 0.0))
        params['tilt_ty'] = float(settings.get('tilt_ty_radpm', 0.0))
        return 'geometric+tilt', params, None

    print(f"[wfe] Unknown SIM_WFE mode '{mode}', skip simulated WFE.")
    return None, None, None

