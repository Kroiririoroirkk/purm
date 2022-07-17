import pickle

from find_calls import OutputEntry, CameraEntry, CallType2

if __name__ == '__main__':
  print('Loading file...')
  with open('output.pickle', 'rb') as f:
    output_arr = pickle.load(f)

  print(output_arr)
