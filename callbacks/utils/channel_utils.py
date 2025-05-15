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

def get_grouped_channels_by_prefix(raw, bad_channels=None):
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

    if 'MEG' in meg_ch_names[int(len(meg_ch_names)/2)]:

        # from file:///home/admin_mel/Downloads/Analysis_of_spontaneous_MEG_activity_in_mild_cogni.pdf
        CHANNEL_GROUPS = {
            'anterior left' : ['MEG 062', 'MEG 063', 'MEG 064', 'MEG 089', 'MEG 090', 'MEG 091', 'MEG 092', 'MEG 093', 'MEG 122', 'MEG 123', 'MEG 124', 'MEG 125', 'MEG 153', 'MEG 177'],
            'anterior right' : ['MEG 061', 'MEG 086', 'MEG 087', 'MEG 088', 'MEG 117', 'MEG 118', 'MEG 119', 'MEG 120', 'MEG 121', 'MEG 149', 'MEG 150', 'MEG 151', 'MEG 152', 'MEG 176', 'MEG 195'],
            'central left' : ['MEG 001', 'MEG 003', 'MEG 005', 'MEG 006', 'MEG 007', 'MEG 008', 'MEG 009', 'MEG 010', 'MEG 011', 'MEG 020', 'MEG 021', 'MEG 022', 'MEG 023', 'MEG 024', 'MEG 025', 'MEG 026', 'MEG 027', 'MEG 028', 'MEG 037', 'MEG 038', 'MEG 039', 'MEG 040', 'MEG 041', 'MEG 042', 'MEG 043', 'MEG 044', 'MEG 045', 'MEG 046', 'MEG 047', 'MEG 048'],
            'central right' : ['MEG 002', 'MEG 004', 'MEG 012', 'MEG 013', 'MEG 014', 'MEG 015', 'MEG 016', 'MEG 017', 'MEG 018', 'MEG 019', 'MEG 029', 'MEG 030', 'MEG 031', 'MEG 032', 'MEG 033', 'MEG 034', 'MEG 035', 'MEG 036', 'MEG 049', 'MEG 050', 'MEG 051', 'MEG 052', 'MEG 053', 'MEG 054', 'MEG 055', 'MEG 056', 'MEG 057', 'MEG 058', 'MEG 059', 'MEG 060'],
            'temporal left' : ['MEG 065', 'MEG 066', 'MEG 067', 'MEG 068', 'MEG 069', 'MEG 070', 'MEG 071', 'MEG 094', 'MEG 095', 'MEG 096', 'MEG 097', 'MEG 098', 'MEG 099', 'MEG 100', 'MEG 101', 'MEG 126', 'MEG 127', 'MEG 128', 'MEG 129', 'MEG 130', 'MEG 131', 'MEG 132', 'MEG 133', 'MEG 154', 'MEG 155', 'MEG 156', 'MEG 157', 'MEG 158', 'MEG 159', 'MEG 160', 'MEG 178', 'MEG 179', 'MEG 180', 'MEG 181', 'MEG 182', 'MEG 196', 'MEG 197', 'MEG 198', 'MEG 199', 'MEG 212', 'MEG 213', 'MEG 214', 'MEG 215', 'MEG 216', 'MEG 229', 'MEG 230', 'MEG 231', 'MEG 232', 'MEG 233', 'MEG 234'],
            'temporal right' : ['MEG 079', 'MEG 080', 'MEG 081', 'MEG 082', 'MEG 083', 'MEG 084', 'MEG 085', 'MEG 109', 'MEG 110', 'MEG 111', 'MEG 112', 'MEG 113', 'MEG 114', 'MEG 115', 'MEG 116', 'MEG 141', 'MEG 142', 'MEG 143', 'MEG 144', 'MEG 145', 'MEG 146', 'MEG 147', 'MEG 148', 'MEG 169', 'MEG 170', 'MEG 171', 'MEG 172', 'MEG 173', 'MEG 174', 'MEG 175', 'MEG 190', 'MEG 191', 'MEG 192', 'MEG 193', 'MEG 194', 'MEG 208', 'MEG 209', 'MEG 210', 'MEG 211', 'MEG 224', 'MEG 225', 'MEG 226', 'MEG 227', 'MEG 228', 'MEG 243', 'MEG 244', 'MEG 245', 'MEG 246', 'MEG 247', 'MEG 248'],
            'posterior left': ['MEG 072', 'MEG 073', 'MEG 074', 'MEG 075', 'MEG 102', 'MEG 103', 'MEG 104', 'MEG 134', 'MEG 135', 'MEG 136', 'MEG 137', 'MEG 161', 'MEG 162', 'MEG 163', 'MEG 164', 'MEG 183', 'MEG 184', 'MEG 185', 'MEG 200', 'MEG 201', 'MEG 202', 'MEG 203', 'MEG 217', 'MEG 218', 'MEG 219', 'MEG 220', 'MEG 235', 'MEG 236', 'MEG 237', 'MEG 238'],
            'posterior right': ['MEG 076', 'MEG 077', 'MEG 078', 'MEG 105', 'MEG 106', 'MEG 107', 'MEG 108', 'MEG 138', 'MEG 139', 'MEG 140', 'MEG 165', 'MEG 166', 'MEG 167', 'MEG 168', 'MEG 186', 'MEG 187', 'MEG 188', 'MEG 189', 'MEG 204', 'MEG 205', 'MEG 206', 'MEG 207', 'MEG 221', 'MEG 222', 'MEG 223', 'MEG 239', 'MEG 240', 'MEG 241', 'MEG 242']
        }
        
        grouped_channels = {}
        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [
                ch for ch in channels if ch in meg_ch_names
            ]
            grouped_channels[region] = filtered_channels

    elif 'M' in meg_ch_names[int(len(meg_ch_names)/2)]:
        for ch_name in meg_ch_names:
            prefix = ch_name.split('-')[0][:3]
            if prefix_pattern.match(prefix):
                grouped_channels[prefix].append(ch_name)
            
        return dict(grouped_channels)

    elif 'A' in meg_ch_names[int(len(meg_ch_names)/2)]:

        CHANNEL_GROUPS = {
            'anterior left': ['A62', 'A63', 'A64', 'A89', 'A90', 'A91', 'A92', 'A93', 'A122', 'A123', 'A124', 'A125', 'A153', 'A177'], 
            'anterior right': ['A61', 'A86', 'A87', 'A88', 'A117', 'A118', 'A119', 'A120', 'A121', 'A149', 'A150', 'A151', 'A152', 'A176', 'A195'], 
            'central left': ['A1', 'A3', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48'], 
            'central right': ['A2', 'A4', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60'], 
            'temporal left': ['A65', 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A94', 'A95', 'A96', 'A97', 'A98', 'A99', 'A100', 'A101', 'A126', 'A127', 'A128', 'A129', 'A130', 'A131', 'A132', 'A133', 'A154', 'A155', 'A156', 'A157', 'A158', 'A159', 'A160', 'A178', 'A179', 'A180', 'A181', 'A182', 'A196', 'A197', 'A198', 'A199', 'A212', 'A213', 'A214', 'A215', 'A216', 'A229', 'A230', 'A231', 'A232', 'A233', 'A234'], 
            'temporal right': ['A79', 'A80', 'A81', 'A82', 'A83', 'A84', 'A85', 'A109', 'A110', 'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A141', 'A142', 'A143', 'A144', 'A145', 'A146', 'A147', 'A148', 'A169', 'A170', 'A171', 'A172', 'A173', 'A174', 'A175', 'A190', 'A191', 'A192', 'A193', 'A194', 'A208', 'A209', 'A210', 'A211', 'A224', 'A225', 'A226', 'A227', 'A228', 'A243', 'A244', 'A245', 'A246', 'A247', 'A248'], 
            'posterior left': ['A72', 'A73', 'A74', 'A75', 'A102', 'A103', 'A104', 'A134', 'A135', 'A136', 'A137', 'A161', 'A162', 'A163', 'A164', 'A183', 'A184', 'A185', 'A200', 'A201', 'A202', 'A203', 'A217', 'A218', 'A219', 'A220', 'A235', 'A236', 'A237', 'A238'], 
            'posterior right': ['A76', 'A77', 'A78', 'A105', 'A106', 'A107', 'A108', 'A138', 'A139', 'A140', 'A165', 'A166', 'A167', 'A168', 'A186', 'A187', 'A188', 'A189', 'A204', 'A205', 'A206', 'A207', 'A221', 'A222', 'A223', 'A239', 'A240', 'A241', 'A242']}
        
        grouped_channels = {}
        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [
                ch for ch in channels if ch in meg_ch_names
            ]
            grouped_channels[region] = filtered_channels

    if bad_channels:
        if isinstance(bad_channels, str):
            bad_channels_list = [ch.strip() for ch in bad_channels.split(",") if ch.strip()]
        else:
            bad_channels_list = list(bad_channels)  # if it's already a list (e.g., from a previous state)
        grouped_channels["bad"]=bad_channels_list

    return grouped_channels


