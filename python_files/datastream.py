import spikeinterface.extractors as se
# recording = se.read_intan("data/RAT3_PRICKING/pricking_170227_133227.rhd", stream_id="0")

# fs = recording.get_sampling_frequency()
# print(f"Sampling frequency: {fs} Hz")

# channels = recording.get_channel_ids()
# print(f"Channels: {channels}")

# data  = recording.get_traces()
# print(f"Data shape: {data.shape}")



# print(recording.get_channel_groups())

# print(recording.get_property_keys())

# Data er 56 kanaler (A000 - A055)30khz og 1 trigger kanal (A-VDD1 ( 1500 samples/s))
# Trigger kanal indeholder 0 og 1, hvor 1 indikerer en trigger
# ca en 60.032s optagelser

# r=se.read_intan(r'data/RAT3_PRICKING/pricking_170227_133227.rhd', stream_id='0')
# n=r.get_num_samples(); fs=r.get_sampling_frequency(); print('duration_s=', n/fs)
# print('num_channels=', r.get_num_channels())
# print('channel_ids=', r.get_channel_ids()); print('properties=', r.get_property_keys())
# print('gains_uV=', r.get_property('gain_to_uV'))
# print('offsets_uV=', r.get_property('offset_to_uV'))
# print('groups=', r.get_property('group'))
# print("channel names=", r.get_property('channel_name'))
# print("group names=", r.get_property('group_name'))
# print("physical unit=", r.get_property('physical_unit'))
# print("gain to physical unit=", r.get_property('gain_to_physical_unit'))
# print("offset to physical unit=", r.get_property('offset_to_physical_unit'))

#RHD2000 amplifier is the chip used

r=se.read_intan(r'data/RAT3_PRICKING/pricking_170227_133227.rhd', stream_id="0")
n=r.get_num_samples(); fs=r.get_sampling_frequency()
print('duration_s=', n/fs); print('num_channels=', r.get_num_channels())
print('channel_ids=', r.get_channel_ids())
print('properties=', r.get_property_keys())
print('gains_uV=', r.get_property('gain_to_uV'))
print('offsets_uV=', r.get_property('offset_to_uV'))
print('groups=', r.get_property('group'))
import neo
p=(r'data/RAT3_PRICKING/pricking_170227_133227.rhd')
r=neo.rawio.IntanRawIO(filename=p); r.parse_header()
print('signal_streams=', r.header['signal_streams'])
print('event_channels=', r.header['event_channels'])
print('signal_channels(sample)=', r.header['signal_channels'][:12])
print("===="*50 + "\n" "traces = " + r.get_traces()  + "\n" + "===="*50)
