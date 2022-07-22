import enum


CHANNELS = {
  'audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav': 8,
  'audio/aviary_2019-06-01_1559399640.000-1559400540.000_audio.wav': 15
}

ROLLS = [0]

KSVD_ITERS = 80

class DatasetType(enum.Enum):
  INIT = ('init', 0.5)
  TRAIN = ('train', 0.5)
  VALIDATE = ('validate', 0)

  def __init__(self, filename, proportion):
    self.filename = filename
    self.proportion = proportion


class CallType(enum.Enum):
  WHISTLE = ('whistle', 0.5, 240, 546, -9, (2000,15000))
  CHATTER = ('chatter', 1.5, 80, 106, -19, (2000,15000))
  BURBLE = ('burble', 0.3, 250, 454, -7, (1500,15000))

  OTHER = ('other', 9999999, 1, 1, 0, (1,1)) # not important

  def __init__(self, filename, duration, sparsity, dict_size, threshold, freq_cutoffs):
    self.filename = filename
    self.duration = duration # in seconds
    self.sparsity = sparsity
    self.dict_size = dict_size
    self.threshold = threshold
    self.freq_cutoffs = freq_cutoffs # in Hz

  @classmethod
  def from_str(cls, s):
    """Outputs a CallType given a string. Throws a KeyError if none is found.

    Keyword arguments:
    s -- the string

    Returns:
    A CallType
    """
    d = {
      'whistle': CallType.WHISTLE,
      'whiste': CallType.WHISTLE,
      'whitsle': CallType.WHISTLE,
      'whsitle': CallType.WHISTLE,
      'w': CallType.WHISTLE,
      'whistle?': CallType.WHISTLE,
      'c': CallType.CHATTER,
      'chatter': CallType.CHATTER,
      'rattle': CallType.CHATTER,
      'b': CallType.BURBLE,
      'burble': CallType.BURBLE,
      's': CallType.OTHER,
      'song': CallType.OTHER,
      'chuck': CallType.OTHER,
      '?': CallType.OTHER
    }
    return d[s]


def get_channel(audio_filename):
  return CHANNELS.get(audio_filename, 0)
