# Sound Detection

The goal of this project is to automate the detection of cowbird calls in audio using dictionary learning. The project is divided into four parts:

1. **Extraction of training data.** Given an audio file and manual annotations, make numerical data to train the algorithm on.
2. **Dictionary generation.** Given training data, generate an appropriate dictionary using the K-SVD algorithm.
3. **Sound detection.** Given an audio file and the appropriate dictionaries, find the timestamps of all calls present in the audio file.
4. **Validation.** Compare the algorithmic output to the manual annotations.

All of the steps are explained in more detail below.

## Extraction of training data

The script `extract_training_data.py` is the main entry point for this step. It reads from the `audio` and `annotations` folder and then saves an array of vectors to the `training_data` folder as a CSV file.

The script `plot_call_durations.py` was used in order to determine the appropriate vector lengths for each type of call.

Before data extraction, the audio is pre-processed using filters from `preprocessing.py`.

The `config.py` file allows the setting of variables that control the process such as the durations of the calls extracted and the amount to which each vector is rolled in the training data.

Finally, spectograms of the existing training data can be viewed using `plot_training_data.py`, which also allows the user to play the training data as audio samples.

## Dictionary generation

The script `gen_dicts.py` is the main entry point for this step. Given the type(s) of call to generate dictionaries for as its command-line argument(s), it reads from the `training_data` folder and then saves the dictionary as a CSV file to the `dictionaries` folder.

Methods from the script `ksvd.py` are used to generate the dictionaries. The `ksvd_tester.py` script was used to validate the accuracy of the K-SVD implementation.

The `config.py` file allows the setting of variables that control the process such as the durations of the calls extracted, the sparsity of the sparse representations generated, and the dictionary length.

Finally, spectograms of the dictionary data can be viewed using `plot_dictionary_data.py`, which also allows the user to play the training data as audio samples.

## Sound detection

The script `detect_sound.py` is the main entry point for this step. Given the audio file and the type of call to detect for as its command-line arguments, it reads from the `dictionaries` folder and saves the timestamps of calls that it detects as well as the signal-to-interference-and-noise ratio to the `output` folder.

The `config.py` file allows the setting of variables that control the process such as the threshold above which the algorithm classifies an audio sample as a call.

Finally, spectograms of the algorithmically detected calls can be viewed using `plot_output_data.py`, which also allows the user to play the training data as audio samples.

## Validation

The script `validation.py` is the main entry point for this step. Given an annotations file and an output file, it compares the two and calculates the sensitivity (true positive rate) and specificity (one minus the false positive rate) of the test, where both are ratios from 0% to 100%.

The script `validate_against_samples.py` is designed to test the detection algorithm against short audio samples which are already annotated. It plots false positives and false negatives as spectograms. Calling it with `noplot` as the last argument suppresses generating the plots.

The `make_roc_curve.py` file allows for convenient generation of receiver operating characteristic curve graphs.
