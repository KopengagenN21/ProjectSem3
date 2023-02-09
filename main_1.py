import cmath
import time
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream

def distance_metric(cov_matrix_1, cov_matrix_2):    # put 2 cov matrices of 2 probes
    return distance

def find_geom_mean(cov_matrices_of_class_1):     # put array of cov matrices
    temp_sum = ...
    cov_matrices_of_class_1[0]
    return result  # cov matrix 30x30


class Riem_classificator:
    def __int__(self, probes_duration, do_filtering=False, freq_of_low_bandpass_border=1, freq_of_high_bandpass_border=150):
        match do_filtering:
            case True:
                self.do_filtering=do_filtering
            case False:

    def get_probe

    def distance_metric:

    def learning_function:

    def classification:


try:

    classificator = Riem_classificator(3, True)



    print("looking for an EEG stream...")
    stream_infos = resolve_stream('type', 'EEG')
    inlet = StreamInlet(stream_infos[0])
    i = 0

    cov_matrices_of_class_1 = []
    cov_matrices_of_class_2 = []

    data_1 = []
    data_2 = []

    count_of_probes = 10
    time_of_probe = 3    # time of 1 probe in seconds

    chunk, t_ = inlet.pull_sample()
    c1 = np.array([chunk])

    fig, ax = plt.subplots()

    for i in range(count_of_probes):
        chunk, t_shtamp = inlet.pull_chunk(timeout=time_of_probe)
        chunk = np.array(chunk).T

        data_1.append(chunk)
        cov_matrices_of_class_1.append(np.cov(chunk, bias=True))

    print ('Need to switch condition...')
    time.sleep(1.5)

    for i in range(count_of_probes):
        chunk, t_shtamp = inlet.pull_chunk(timeout=time_of_probe)
        chunk = np.array(chunk).T

        data_2.append(chunk)
        cov_matrices_of_class_2.append(np.cov(chunk, bias=True))

    ax.plot(data_1[0].T, linewidth=2)
    plt.show()



    # Place for algorithm's learning

    count_of_probes_for_classification = 10
    geom_mean_of_class_1 = np.zeros((cov_matrices_of_class_1[0].shape[0]))
    geom_mean_of_class_2 = np.zeros((cov_matrices_of_class_1[0].shape[0]))
    result_of_classification = ''
    for i in range(count_of_probes_for_classification):
        chunk, t_shtamp = inlet.pull_chunk(timeout=time_of_probe)
        chunk = np.array(chunk).T

        cov_mat = np.cov(chunk, bias=True)
        if distance_metric(cov_mat, geom_mean_of_class_1) < distance_metric(cov_mat, geom_mean_of_class_2):
            result_of_classification = 'belong 1st class'
        else:
            result_of_classification = 'belong 2nd class'

        print('This probe ', result_of_classification)



    #
    # while i < 719:
    #     chunk, t_ = inlet.pull_sample()
    #     c1 = np.vstack((c1, [chunk]))
    #     i = i + 1
    # print("c1\n", c1, "\n")
    #
    # chunk, t_ = inlet.pull_sample()
    # c2 = np.array([chunk])
    # # print("c2\n", c2, "\n")
    # # print("\n")
    #
    # i = 0
    # while i < 719:
    #     chunk, t_ = inlet.pull_sample()
    #     c2 = np.vstack((c2, [chunk]))
    #     # print(t_, chunk)
    #     i = i + 1
    #
    # print("c2\n", c2, "\n")
    #
    # print(c1.shape, c2.shape)

    cov1 = np.cov(c1.T, bias=True)
    cov2 = np.cov(c2.T, bias=True)
    # print("cov1\n", cov1, "\n\ncov2\n", cov2)
    print (cov1.shape, cov2.shape)

    c1_ObrMatrix = np.linalg.inv(cov1)
    # print("\n\nc1_ObrMatrix\n", c1_ObrMatrix, "\n")

    multiplication = np.matmul(c1_ObrMatrix, cov2)
    # print("\nmultiplication\n", multiplication, "\n")

    eigenvalues, v = np.linalg.eig(multiplication)
    # print("\neigenvalues\n", eigenvalues, "\n")

    size = len(eigenvalues)
    # print("\nsize\n", size, "\n")

    i = 0
    sum = 0
    while i < size:
        #res = cmath.phase(eigenvalues[i])
        sum = sum + cmath.log(eigenvalues[i]) ** 2
        i = i + 1
        #print(sum)

    result = sum ** 0.5
    print("\nsum\n", result, "\n")

except KeyboardInterrupt as e:
    print("Ending program")
raise e
