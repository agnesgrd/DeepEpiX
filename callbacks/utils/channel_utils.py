import re
from collections import defaultdict
import mne

# def remove_leading_zeros_from_channels(channel_groups):
#     updated_groups = {}
#     for region, channels in channel_groups.items():
#         updated_channels = []
#         for ch in channels:
#             # Strip leading zeros from numeric part, e.g. 'A001' -> 'A1'
#             if ch.startswith('A') and ch[1:].isdigit():
#                 num = str(int(ch[1:]))  # convert to int and back to string to remove zeros
#                 updated_channels.append(f'A{num}')
#             else:
#                 updated_channels.append(ch)  # in case of unexpected format
#         updated_groups[region] = updated_channels
#     return updated_groups

def get_grouped_channels_by_prefix(raw):
    """
    Load channels from raw data and group them by their 3-letter prefix.

    Returns:
        dict: Dictionary where keys are 3-letter prefixes and values are lists of channel names.
    """
    grouped_channels = defaultdict(list)
    prefix_pattern = re.compile(r'^[A-Z]{3}$')

    # Get only MEG channels (both magnetometers and gradiometers)
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
    meg_ch_names = [raw.info['ch_names'][i] for i in meg_picks]

    if 'M' in meg_ch_names[int(len(meg_ch_names)/2)]:
        for ch_name in meg_ch_names:
            prefix = ch_name.split('-')[0][:3]
            if prefix_pattern.match(prefix):
                grouped_channels[prefix].append(ch_name)
            
        return dict(grouped_channels)

    elif 'A' in meg_ch_names[int(len(meg_ch_names)/2)]:

        # from file:///home/admin_mel/Downloads/Analysis_of_spontaneous_MEG_activity_in_mild_cogni.pdf
        # CHANNEL_GROUPS = {
        #     "anterior": [
        #         'A072', 'A073', 'A050', 'A070', 'A094', 'A093'
        #     ],
        #     "central": [
        #         'A113', 'A095', 'A074', 'A052', 'A031', 'A092', 'A051', 'A030', 'A032',
        #         'A075', 'A053', 'A015', 'A014', 'A091', 'A033', 'A034', 'A004', 'A013',
        #         'A028', 'A029', 'A047', 'A054', 'A035', 'A017', 'A005', 'A012', 'A027',
        #         'A067', 'A076', 'A055', 'A018', 'A006', 'A011', 'A026', 'A066', 'A089',
        #         'A077', 'A056', 'A036', 'A019', 'A007', 'A010', 'A025', 'A065', 'A088',
        #         'A078', 'A057', 'A037', 'A020', 'A008', 'A009', 'A024', 'A064', 'A087',
        #         'A079', 'A021', 'A022', 'A023', 'A043', 'A044', 'A063', 'A086'
        #     ],
        #     "left_temporal": [
        #         'A131', 'A114', 'A096', 'A097', 'A098', 'A099', 'A100',
        #         'A115', 'A116', 'A117', 'A118',
        #         'A132', 'A133', 'A134', 'A135', 'A136', 'A137'
        #     ],
        #     "right_temporal": [
        #         'A130', 'A112', 'A111', 'A110', 'A109', 'A108', 'A107',
        #         'A128', 'A127', 'A126', 'A125', 'A124', 'A123',
        #         'A147', 'A146', 'A145', 'A144', 'A143', 'A142', 'A141'
        #     ],
        #     "posterior": [
        #         'A038', 'A058', 'A039', 'A040', 'A041', 'A060', 'A061',
        #         'A081', 'A082', 'A083', 'A084', 'A085', 'A106',
        #         'A101', 'A102', 'A104', 'A105',
        #         'A119', 'A120', 'A121', 'A122', 'A123', 'A124',
        #         'A137', 'A138', 'A139', 'A140', 'A141'
        #     ]
        # }

        CHANNEL_GROUPS = {
            'anterior': ['A30', 'A31', 'A32', 'A48', 'A49', 'A50', 'A51', 'A52', 'A69', 'A70', 'A71', 'A72', 'A73', 'A74', 'A92', 'A93', 'A94'],

            'central': [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
            'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
            'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29'],

            'left_temporal': [
                'A33', 'A34', 'A35', 'A36', 'A37',
                'A53', 'A54', 'A55', 'A56', 'A57',
                'A75', 'A76', 'A77', 'A78', 'A79', 'A80',
                'A95', 'A96', 'A97', 'A98', 'A99', 'A100',
                'A113', 'A114', 'A115', 'A116', 'A117', 'A118'],

            'right_temporal': [
                'A43', 'A44', 'A45', 'A46', 'A47', 
                'A64', 'A65', 'A66', 'A67', 'A68', 
                'A86', 'A87', 'A88', 'A89', 'A90', 'A91', 
                'A107', 'A108', 'A109', 'A110', 'A111', 'A112', 
                'A125', 'A126', 'A127', 'A128', 'A129', 'A130'], 

            'posterior': [
                'A38', 'A39', 'A40', 'A41', 'A42', 
                'A58', 'A59', 'A60', 'A61', 'A62', 'A63', 
                'A81', 'A82', 'A83', 'A84', 'A85', 
                'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 
                'A119', 'A120', 'A121', 'A122', 'A123', 'A124'],
            
            'exterior': [
                'A131', 'A132', 'A133', 'A134', 'A135', 'A136', 'A137', 'A138', 'A139', 
                'A140', 'A141', 'A142', 'A143', 'A144', 'A145', 'A146', 'A147', 'A148']
            }
        
        filtered_groups = {}
        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [
                ch for ch in channels if ch in meg_ch_names
            ]
            filtered_groups[region] = filtered_channels

        return filtered_groups


