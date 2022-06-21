import enum


ROLLS = [0]

KSVD_ITERS = 50

FREQ_CUTOFF = (2000, 15000)


class DatasetType(enum.Enum):
  INIT = ('init', 0.4)
  TRAIN = ('train', 0.4)
  VALIDATE = ('validate', 0.2)

  def __init__(self, filename, proportion):
    self.filename = filename
    self.proportion = proportion


class CallType(enum.Enum):
  WHISTLE = ('whistle', 0.5, 30, 362, -15.5)
  CHATTER = ('chatter', 1.5, 30, 60, -24)
  BURBLE = ('burble', 0.2, 2, 3, -9)

  SONG = ('song', 1.3, 2, 5, -9) # these two are not terribly important
  CHUCK = ('chuck', 0.18, 2, 5, -9)

  def __init__(self, filename, duration, sparsity, dict_size, threshold):
    self.filename = filename
    self.duration = duration # in seconds
    self.sparsity = sparsity
    self.dict_size = dict_size
    self.threshold = threshold

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
    }
    return d[s]

