import enum


CHANNELS = {
  'audio/aviary_2019-05-01_1556722860.000-1556723760.000_audio.wav': 8,
  'audio/aviary_2019-06-01_1559399640.000-1559400540.000_audio.wav': 15
}

ROLLS = [0]

KSVD_ITERS = 50

class DatasetType(enum.Enum):
  INIT = ('init', 0.4)
  TRAIN = ('train', 0.4)
  VALIDATE = ('validate', 0.2)

  def __init__(self, filename, proportion):
    self.filename = filename
    self.proportion = proportion


class CallType(enum.Enum):
  WHISTLE = ('whistle', 0.5, 30, 233, -17, (2000,15000))
  CHATTER = ('chatter', 1.5, 30, 60, -24, (2000,15000))
  BURBLE = ('burble', 0.2, 40, 59, -12, (2500,12000))

  SONG = ('song', 1.3, 0, 0, 0, (1,1)) # these three are not terribly important
  CHUCK = ('chuck', 0.18, 0, 0, 0, (1,1))
  UNKNOWN = ('unknown', 0.5, 0, 0, 0, (1,1))

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
      's': CallType.SONG,
      'song': CallType.SONG,
      'c': CallType.CHATTER,
      'chatter': CallType.CHATTER,
      'rattle': CallType.CHATTER,
      'b': CallType.BURBLE,
      'burble': CallType.BURBLE,
      'chuck': CallType.CHUCK,
      '?': CallType.UNKNOWN
    }
    return d[s]


def get_channel(audio_filename):
  return CHANNELS.get(audio_filename, 0)
