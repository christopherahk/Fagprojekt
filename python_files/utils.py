"""
rhd2000_utils.py

Utility module for reading Intan Technologies RHD2000 data files (.rhd).
Designed to be imported from a Jupyter notebook or other Python script.

Example (in a .ipynb):
    from rhd2000_utils import read_rhd, summarize

    data = read_rhd('path/to/recording.rhd')
    summarize(data)

    import matplotlib.pyplot as plt
    plt.plot(data['t_amplifier'], data['amplifier_data'][0])
    plt.show()
"""

import struct
import os
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FMT = {
    'uint32': ('<I', 4),
    'int16':  ('<h', 2),
    'uint16': ('<H', 2),
    'int32':  ('<i', 4),
    'single': ('<f', 4),
}

def _fread(fid, dtype):
    """Read a single scalar value from an open binary file."""
    fmt, size = _FMT[dtype]
    return struct.unpack(fmt, fid.read(size))[0]


def _read_qstring(fid):
    """
    Read a Qt-style QString.
    The leading uint32 gives byte length; 0xFFFFFFFF means null/empty string.
    Characters are UTF-16LE (2 bytes each).
    """
    raw = fid.read(4)
    if len(raw) < 4:
        return ''
    length = struct.unpack('<I', raw)[0]
    if length == 0xFFFFFFFF:
        return ''
    return ''.join(chr(struct.unpack('<H', fid.read(2))[0]) for _ in range(length // 2))


def _notch_filter(signal, f_sample, f_notch, bandwidth):
    """
    IIR notch filter (e.g. 50 or 60 Hz).

    Parameters
    ----------
    signal    : 1-D numpy array
    f_sample  : sample rate in Hz
    f_notch   : notch centre frequency in Hz
    bandwidth : 3-dB bandwidth in Hz (10 Hz recommended)
    """
    tstep = 1.0 / f_sample
    Fc    = f_notch * tstep
    d     = np.exp(-2.0 * np.pi * (bandwidth / 2.0) * tstep)
    b     = (1.0 + d * d) * np.cos(2.0 * np.pi * Fc)
    a0, a1, a2 = 1.0, -b, d * d
    a    = (1.0 + d * d) / 2.0
    b1   = -2.0 * np.cos(2.0 * np.pi * Fc)

    L   = len(signal)
    out = np.zeros(L)
    out[0] = signal[0]
    if L > 1:
        out[1] = signal[1]
    for i in range(2, L):
        out[i] = (a * signal[i] + a * b1 * signal[i-1] + a * signal[i-2]
                  - a1 * out[i-1] - a2 * out[i-2]) / a0
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_rhd(filepath):
    """
    Read an Intan RHD2000 data file and return all data as a dictionary.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to a .rhd file.

    Returns
    -------
    data : dict
        Keys include:
          Header / metadata
            'notes', 'frequency_parameters', 'reference_channel'
          Channel descriptors  (list of dicts, one per channel)
            'amplifier_channels', 'spike_triggers',
            'aux_input_channels', 'supply_voltage_channels',
            'board_adc_channels', 'board_dig_in_channels',
            'board_dig_out_channels'
          Data arrays  (numpy arrays, present only when the file contains data)
            'amplifier_data'      shape (n_channels, n_samples)  – µV
            't_amplifier'         shape (n_samples,)              – seconds
            'aux_input_data'      shape (n_ch, n_samples)         – V
            't_aux_input'         shape (n_samples,)              – seconds
            'supply_voltage_data' shape (n_ch, n_blocks)          – V
            't_supply_voltage'    shape (n_blocks,)               – seconds
            'board_adc_data'      shape (n_ch, n_samples)         – V
            't_board_adc'         shape (n_samples,)              – seconds
            'board_dig_in_data'   shape (n_ch, n_samples)         – bool
            'board_dig_out_data'  shape (n_ch, n_samples)         – bool
            't_dig'               shape (n_samples,)              – seconds
            'temp_sensor_data'    shape (n_ch, n_blocks)          – °C
            't_temp_sensor'       shape (n_blocks,)               – seconds

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the file does not start with the RHD2000 magic number.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'File not found: {filepath}')

    filesize = os.path.getsize(filepath)

    with open(filepath, 'rb') as fid:

        # Magic number
        if _fread(fid, 'uint32') != 0xC6912702:
            raise ValueError('Not a valid RHD2000 file (magic number mismatch).')

        # Version
        main_ver = _fread(fid, 'int16')
        sec_ver  = _fread(fid, 'int16')

        spb = 60 if main_ver == 1 else 128  # samples per data block

        # Sampling / filter settings
        sample_rate                  = _fread(fid, 'single')
        dsp_enabled                  = _fread(fid, 'int16')
        actual_dsp_cutoff_frequency  = _fread(fid, 'single')
        actual_lower_bandwidth       = _fread(fid, 'single')
        actual_upper_bandwidth       = _fread(fid, 'single')
        desired_dsp_cutoff_frequency = _fread(fid, 'single')
        desired_lower_bandwidth      = _fread(fid, 'single')
        desired_upper_bandwidth      = _fread(fid, 'single')

        notch_mode      = _fread(fid, 'int16')
        notch_freq      = {0: 0, 1: 50, 2: 60}.get(notch_mode, 0)

        desired_impedance_test_frequency = _fread(fid, 'single')
        actual_impedance_test_frequency  = _fread(fid, 'single')

        notes = {
            'note1': _read_qstring(fid),
            'note2': _read_qstring(fid),
            'note3': _read_qstring(fid),
        }

        num_temp_sensor_channels = 0
        if (main_ver == 1 and sec_ver >= 1) or main_ver > 1:
            num_temp_sensor_channels = _fread(fid, 'int16')

        eval_board_mode = 0
        if (main_ver == 1 and sec_ver >= 3) or main_ver > 1:
            eval_board_mode = _fread(fid, 'int16')

        reference_channel = _read_qstring(fid) if main_ver > 1 else ''

        frequency_parameters = {
            'amplifier_sample_rate':            sample_rate,
            'aux_input_sample_rate':            sample_rate / 4,
            'supply_voltage_sample_rate':       sample_rate / spb,
            'board_adc_sample_rate':            sample_rate,
            'board_dig_in_sample_rate':         sample_rate,
            'desired_dsp_cutoff_frequency':     desired_dsp_cutoff_frequency,
            'actual_dsp_cutoff_frequency':      actual_dsp_cutoff_frequency,
            'dsp_enabled':                      dsp_enabled,
            'desired_lower_bandwidth':          desired_lower_bandwidth,
            'actual_lower_bandwidth':           actual_lower_bandwidth,
            'desired_upper_bandwidth':          desired_upper_bandwidth,
            'actual_upper_bandwidth':           actual_upper_bandwidth,
            'notch_filter_frequency':           notch_freq,
            'desired_impedance_test_frequency': desired_impedance_test_frequency,
            'actual_impedance_test_frequency':  actual_impedance_test_frequency,
        }

        # Channel lists
        amplifier_channels      = []
        aux_input_channels      = []
        supply_voltage_channels = []
        board_adc_channels      = []
        board_dig_in_channels   = []
        board_dig_out_channels  = []
        spike_triggers          = []

        n_groups = _fread(fid, 'int16')
        for group_idx in range(n_groups):
            grp_name    = _read_qstring(fid)
            grp_prefix  = _read_qstring(fid)
            grp_enabled = _fread(fid, 'int16')
            grp_n_ch    = _fread(fid, 'int16')
            _           = _fread(fid, 'int16')  # num_amp_channels (unused)

            if grp_n_ch > 0 and grp_enabled > 0:
                for _ in range(grp_n_ch):
                    native_name  = _read_qstring(fid)
                    custom_name  = _read_qstring(fid)
                    native_order = _fread(fid, 'int16')
                    custom_order = _fread(fid, 'int16')
                    sig_type     = _fread(fid, 'int16')
                    ch_enabled   = _fread(fid, 'int16')
                    chip_ch      = _fread(fid, 'int16')
                    board_stream = _fread(fid, 'int16')
                    v_trig_mode  = _fread(fid, 'int16')
                    v_threshold  = _fread(fid, 'int16')
                    dig_trig_ch  = _fread(fid, 'int16')
                    dig_edge_pol = _fread(fid, 'int16')
                    imp_mag      = _fread(fid, 'single')
                    imp_phase    = _fread(fid, 'single')

                    if not ch_enabled:
                        continue

                    ch = {
                        'native_channel_name':           native_name,
                        'custom_channel_name':           custom_name,
                        'native_order':                  native_order,
                        'custom_order':                  custom_order,
                        'board_stream':                  board_stream,
                        'chip_channel':                  chip_ch,
                        'port_name':                     grp_name,
                        'port_prefix':                   grp_prefix,
                        'port_number':                   group_idx,
                        'electrode_impedance_magnitude': imp_mag,
                        'electrode_impedance_phase':     imp_phase,
                    }
                    trig = {
                        'voltage_trigger_mode':    v_trig_mode,
                        'voltage_threshold':       v_threshold,
                        'digital_trigger_channel': dig_trig_ch,
                        'digital_edge_polarity':   dig_edge_pol,
                    }

                    targets = {
                        0: (amplifier_channels, spike_triggers, trig),
                        1: (aux_input_channels,),
                        2: (supply_voltage_channels,),
                        3: (board_adc_channels,),
                        4: (board_dig_in_channels,),
                        5: (board_dig_out_channels,),
                    }
                    if sig_type not in targets:
                        raise ValueError(f'Unknown signal type: {sig_type}')
                    targets[sig_type][0].append(ch)
                    if sig_type == 0:
                        spike_triggers.append(trig)

        n_amp  = len(amplifier_channels)
        n_aux  = len(aux_input_channels)
        n_sv   = len(supply_voltage_channels)
        n_adc  = len(board_adc_channels)
        n_din  = len(board_dig_in_channels)
        n_dout = len(board_dig_out_channels)
        n_temp = num_temp_sensor_channels

        # Bytes per data block
        bpb = spb * 4                       # timestamps
        bpb += spb * 2 * n_amp
        bpb += (spb // 4) * 2 * n_aux
        bpb += 2 * n_sv
        bpb += spb * 2 * n_adc
        if n_din  > 0: bpb += spb * 2
        if n_dout > 0: bpb += spb * 2
        if n_temp > 0: bpb += 2 * n_temp

        bytes_remaining = filesize - fid.tell()
        data_present    = bytes_remaining > 0
        n_blocks        = bytes_remaining // bpb if bpb else 0

        n_amp_samp  = spb * n_blocks
        n_aux_samp  = (spb // 4) * n_blocks
        n_sv_samp   = n_blocks
        n_adc_samp  = spb * n_blocks
        n_din_samp  = spb * n_blocks
        n_dout_samp = spb * n_blocks

        record_time = n_amp_samp / sample_rate if sample_rate else 0

        # Pre-allocate
        t_amplifier         = np.zeros(n_amp_samp,        dtype=np.int32)
        amplifier_data      = np.zeros((n_amp,  n_amp_samp),  dtype=np.uint16)
        aux_input_data      = np.zeros((n_aux,  n_aux_samp),  dtype=np.uint16)
        supply_voltage_data = np.zeros((n_sv,   n_sv_samp),   dtype=np.uint16)
        temp_sensor_data    = np.zeros((n_temp, n_sv_samp),   dtype=np.int16)
        board_adc_data      = np.zeros((n_adc,  n_adc_samp),  dtype=np.uint16)
        board_dig_in_raw    = np.zeros(n_din_samp,  dtype=np.uint16)
        board_dig_out_raw   = np.zeros(n_dout_samp, dtype=np.uint16)

        # Read data blocks
        if data_present:
            ai = xi = si = di = ii = oi = 0

            for blk in range(n_blocks):
                # Timestamps
                signed = (main_ver == 1 and sec_ver >= 2) or main_ver > 1
                ts = np.frombuffer(fid.read(spb * 4), dtype='<i4' if signed else '<u4')
                t_amplifier[ai:ai + spb] = ts.astype(np.int32)

                if n_amp > 0:
                    raw = np.frombuffer(fid.read(spb * 2 * n_amp), dtype='<u2')
                    amplifier_data[:, ai:ai + spb] = raw.reshape(spb, n_amp).T

                aux_spb = spb // 4
                if n_aux > 0:
                    raw = np.frombuffer(fid.read(aux_spb * 2 * n_aux), dtype='<u2')
                    aux_input_data[:, xi:xi + aux_spb] = raw.reshape(aux_spb, n_aux).T

                if n_sv > 0:
                    supply_voltage_data[:, si] = np.frombuffer(fid.read(2 * n_sv), dtype='<u2')

                if n_temp > 0:
                    temp_sensor_data[:, si] = np.frombuffer(fid.read(2 * n_temp), dtype='<i2')

                if n_adc > 0:
                    raw = np.frombuffer(fid.read(spb * 2 * n_adc), dtype='<u2')
                    board_adc_data[:, di:di + spb] = raw.reshape(spb, n_adc).T

                if n_din > 0:
                    board_dig_in_raw[ii:ii + spb] = np.frombuffer(fid.read(spb * 2), dtype='<u2')

                if n_dout > 0:
                    board_dig_out_raw[oi:oi + spb] = np.frombuffer(fid.read(spb * 2), dtype='<u2')

                ai += spb;  xi += aux_spb;  si += 1
                di += spb;  ii += spb;      oi += spb

    # Post-processing (file is now closed)
    if data_present:

        # Unpack digital bitmasks
        board_dig_in_data  = np.zeros((n_din,  n_din_samp),  dtype=bool)
        board_dig_out_data = np.zeros((n_dout, n_dout_samp), dtype=bool)
        for i, ch in enumerate(board_dig_in_channels):
            board_dig_in_data[i]  = (board_dig_in_raw  & (1 << ch['native_order'])) > 0
        for i, ch in enumerate(board_dig_out_channels):
            board_dig_out_data[i] = (board_dig_out_raw & (1 << ch['native_order'])) > 0

        # Scale to physical units
        amplifier_data      = 0.195  * (amplifier_data.astype(np.float64) - 32768)  # µV
        aux_input_data      = 37.4e-6 * aux_input_data.astype(np.float64)            # V
        supply_voltage_data = 74.8e-6 * supply_voltage_data.astype(np.float64)       # V
        temp_sensor_data    = temp_sensor_data.astype(np.float64) / 100.0            # °C

        adc_f64 = board_adc_data.astype(np.float64)
        if eval_board_mode == 1:
            board_adc_data = 152.59e-6 * (adc_f64 - 32768)
        elif eval_board_mode == 13:
            board_adc_data = 312.5e-6  * (adc_f64 - 32768)
        else:
            board_adc_data = 50.354e-6 *  adc_f64

        # Timestamp gap check
        n_gaps = int(np.sum(np.diff(t_amplifier) != 1))
        if n_gaps > 0:
            warnings.warn(f'{n_gaps} gap(s) in timestamp data — time axis will not be uniform!')

        # Time axes (seconds)
        t_amplifier      = t_amplifier / sample_rate
        t_aux_input      = t_amplifier[::4]
        t_supply_voltage = t_amplifier[::spb]
        t_board_adc      = t_amplifier
        t_dig            = t_amplifier
        t_temp_sensor    = t_supply_voltage

        # Notch filter
        if notch_freq > 0:
            for i in range(n_amp):
                amplifier_data[i] = _notch_filter(amplifier_data[i], sample_rate, notch_freq, 10)

    # Assemble output dict
    result = {
        'notes':                    notes,
        'frequency_parameters':     frequency_parameters,
        'reference_channel':        reference_channel,
        'amplifier_channels':       amplifier_channels,
        'spike_triggers':           spike_triggers,
        'aux_input_channels':       aux_input_channels,
        'supply_voltage_channels':  supply_voltage_channels,
        'board_adc_channels':       board_adc_channels,
        'board_dig_in_channels':    board_dig_in_channels,
        'board_dig_out_channels':   board_dig_out_channels,
    }

    if data_present:
        result.update({
            'amplifier_data':       amplifier_data,
            't_amplifier':          t_amplifier,
            'aux_input_data':       aux_input_data,
            't_aux_input':          t_aux_input,
            'supply_voltage_data':  supply_voltage_data,
            't_supply_voltage':     t_supply_voltage,
            'board_adc_data':       board_adc_data,
            't_board_adc':          t_board_adc,
            'board_dig_in_data':    board_dig_in_data,
            'board_dig_out_data':   board_dig_out_data,
            't_dig':                t_dig,
            'temp_sensor_data':     temp_sensor_data,
            't_temp_sensor':        t_temp_sensor,
        })

    return result


def summarize(data):
    """
    Print a structured overview of the dictionary returned by read_rhd().

    Parameters
    ----------
    data : dict
        The dictionary returned by read_rhd().
    """
    if not data:
        print('(empty)')
        return

    def _detail(value):
        if isinstance(value, np.ndarray):
            return f'shape={value.shape}  dtype={value.dtype}'
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                return f'{len(value)} channels  keys={list(value[0].keys())}'
            return f'{len(value)} items'
        if isinstance(value, dict):
            return f'{len(value)} keys  {list(value.keys())}'
        if isinstance(value, str):
            return repr(value) if len(value) < 60 else repr(value[:57] + '...')
        return repr(value)

    c1, c2 = 26, 10
    print(f"{'Key':<{c1}} {'Type':<{c2}} Shape / Value")
    print('-' * 100)
    for key, value in data.items():
        print(f'{key:<{c1}} {type(value).__name__:<{c2}} {_detail(value)}')
