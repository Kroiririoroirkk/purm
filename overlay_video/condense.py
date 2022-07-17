import json
from readin import N, View, get_annotations

json_file = {
  'project': {
    'pid': '__VIA_PROJECT_ID__',
    'rev': '__VIA_PROJECT_REV_ID__',
    'rev_timestamp': '__VIA_PROJECT_REV_TIMESTAMP__',
    'pname': 'Aviary Video Annotation',
    'creator': 'VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)',
    'created': 1642086744960, # Arbitrary
    'data_format_version': '3.1.1',
    'vid_list': ['1', '2']
  },
  'config': {
    'file': {
      'loc_prefix': {
        '1': '',
        '2': '',
        '3': '',
        '4': ''
      }
    },
    'ui': {
      'file_content_align': 'center',
      'gtimeline_visible_row_count': '3',
      'file_metadata_editor_visible': True,
      'spatial_metadata_editor_visible': True,
      'temporal_segment_metadata_editor_visible': True,
      'spatial_region_label_attribute_id': ''
    }
  },
  'attribute': {
    '1': {
      'aname': 'Behavior',
      'anchor_id': 'FILE1_Z2_XY0',
      'type': 3,
      'desc': 'Type of Vocalization',
      'options': {
        '0': 'Song',
        '1': 'Chatter'
      },
      'default_option_id': ''
    },
    '2': {
      'aname': 'Bird',
      'anchor_id': 'FILE1_Z1_XY1',
      'type': 3,
      'desc': 'Select marker type',
      'options': {
        '0': 'StartSitOrStand',
        '1': 'Middle',
        '2': 'EndSitOrStand'
      },
      'default_option_id': '1'
    }
  },
  'file': {
    '1': {
      'fid': '1',
      'fname': 'top.mp4',
      'type': 4,
      'loc': 1,
      'src': ''
    },
    '2': {
      'fid': '2',
      'fname': 'bottom.mp4',
      'type': 4,
      'loc': 1,
      'src': ''
    }
  },
  'view': {
    '1': {
      'fid_list': ['1']
    },
    '2': {
      'fid_list': ['2']
    }
  },
  'metadata': {}
}

song_annotations = get_annotations()[1]

for bird_id in range(N):
  for view, song_entries in song_annotations[bird_id+1].items():
    for song_entry in song_entries:
      json_obj = {
        'vid': 1 if view == View.TOP else 2,
        'flg': 0,
        'z': [song_entry.start_time, song_entry.end_time],
        'xy': [],
        'av': {'1': f'{bird_id+1} {song_entry.song_type.value}'}
      }
      json_file['metadata'][song_entry.uid] = json_obj

json_str = json.dumps(json_file)

with open('compact_json.json', 'w') as f:
  f.write(json_str)

