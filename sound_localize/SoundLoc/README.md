# SoundLoc

SRP-PHAT sound localization for UPenn Aviary

## Installation

You will need Python3, Numpy, Scipy and gcc.
(Note that gcc isn't a python module, it installs to Ubuntu)

You will have to compile soundloc.c. This can be done on most machines by calling
```
python3 setup.py
```
Ammon did this using Ubuntu 18.04.6:
```
python3 setup.py build
sudo python3 setup.py install
```
## Usage

The main program can be run by calling
```
python3 SoundLoc.py <mic_locs.txt> <cage_dims.txt> <song.wav> <bird_locs.txt>
```
The result prints out the point with the detected maximum srp and the identity of the bird nearest this point (in the order that the bird locations are listed in bird_locs.txt e.g. Singer: 0 means the first bird of bird_locs.txt).

There is an example script in the Example directory.

Line 137 of SoundLoc.py can be uncommented (and 138, commented) if you want to use block processing. There is no principled way to choose block and hop sizes currently included here so you must set them yourself.

## WARNING

This program calls a backend C function in line 100 of SoundLoc.py. The C code is not "production" ready--it doesn't properly check inputs. Be careful if you change anything that gets stuck into the \_filterAndSum function.

The C backend function performs the same function as the commented code starting on line 102 in SoundLoc.py. The problem with the pure python code is that an intermediate array needs to be created in order to use einsum. Typically, this takes up too much memory; if you do any block processing with a typical song.wav expect it to want more than 50 GB of memory. The pure python alternative was to add nested for loops but this took way too long to execute.
