import enum


ROLLS = [-1/8, -1/16, 0, 1/16, 1/8]

KSVD_ITERS = 50


class CallType(enum.Enum):
  WHISTLE = ('whistle', 0.6, 3, 50, -33) # modify sparsity, dict_size, and threshold later, used to be 0.7 s
  SONG = ('song', 1.3, 2, 5, -9)
  CHATTER = ('chatter', 2.5, 3, 30, -9)
  BURBLE = ('burble', 0.8, 2, 3, -9)
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
      'burble': CallType.BURBLE,
      'chuck': CallType.CHUCK,
    }
    return d[s]

