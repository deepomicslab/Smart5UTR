import numpy as np
from .model import split_autoencoder
from .dataloader import binary_mtx, decode_seq, onehot_singleseq

def distort(decoder, oldseq, high_neck, decoded_input_len=80, coef=1.2, coef_repeat_num = 20):
    """
    Distort the input sequence by generating a new sequence using the decoder.

    Args:
        decoder: The decoder part of the autoencoder model.
        oldseq: The input sequence.
        high_neck: The high-dimensional representation of the input sequence.
        decoded_input_len: The input length for the decoder (default: 80).
        coef: The coefficient used to adjust the high_neck vector (default: 1.2).
        coef_repeat_num: The repeat number of the coefficient (default: 20).

    Returns:
        newseq: The distorted sequence.
    """
    distimes = 0  ## record distort times
    newseq = ''

    while ((newseq == '') or (newseq == oldseq) or ('atg' in newseq)):
        distimes += 1
        new_high_neck = high_neck * \
                        (np.append((coef + np.random.normal(0, 1, (1, decoded_input_len - coef_repeat_num)) / 4), [coef] * coef_repeat_num))
        decodedmtx = decoder.predict(new_high_neck)  ## numeric matrix
        newseq = decode_seq(binary_mtx(decodedmtx)[0])  ## string

        if (distimes > 1000):
            return ''

    return newseq

def iter_ae(autoencoder, scaler, encoder, decoder, rawseq, coef = 1.2, decoded_input_len=80, iterations=10):
    """
    Iterate and optimize the input sequence using the autoencoder model.

    Args:
        autoencoder: The pre-trained autoencoder model.
        scaler: The scaler used for scaling the output values.
        encoder: The encoder part of the autoencoder model.
        decoder: The decoder part of the autoencoder model.
        rawseq: The input sequence to be optimized.
        decoded_input_len: The input length for the decoder (default: 80).
        iterations: The number of iterations for the optimization (default: 10).

    Returns:
        recorded_seqs: A NumPy array containing the optimized sequences and their output values.
    """
    high_neck = encoder.predict(onehot_singleseq(rawseq))  ## start with latent neck vector
    newrl = 0  ## initial value
    oldrl = autoencoder.predict(onehot_singleseq(rawseq))[1]   ## start with the output value
    oldrl = scaler.inverse_transform(oldrl).reshape(-1)[0]
    oldseq = rawseq
    newseq = rawseq  ## initial value
    update_times = 0
    coef = coef
    max_iters = 10000

    recorded_seqs = np.empty((iterations + 1, 2), dtype=object)
    recorded_seqs[0, 0] = oldseq
    recorded_seqs[0, 1] = oldrl

    while(update_times < iterations):

        update_times += 1
        random_times = 0

        while(newrl <= oldrl):   ## redo distort
            random_times += 1
            newseq = distort(decoder, oldseq, high_neck, decoded_input_len, coef)
            if (newseq == ''):
                print("[Early stop] seq can not be updated through 1000 times.")
                return 0

            newrl = autoencoder.predict(onehot_singleseq(newseq))[1]
            newrl = scaler.inverse_transform(newrl).reshape(-1)[0]

            if (update_times == 1) & (random_times > 5000):  ## bad starting point
                print("[Early stop] model can not generate a better seq at the first iteration.")
                return 0

            if (random_times > max_iters):
                print("Warning: iterations exceed max value, stop evolving on this seq.")
                print(np.where(recorded_seqs[:, 0] == None))
                recorded_seqs[:, 0][np.where(recorded_seqs[:, 0] == None)] = oldseq
                recorded_seqs[:, 1][np.where(recorded_seqs[:, 1] == None)] = oldrl ## copy the last valid value
                return recorded_seqs

        ## update bottle neck vector
        high_neck = encoder.predict(onehot_singleseq(newseq))

        recorded_seqs[update_times, 0] = newseq
        recorded_seqs[update_times, 1] = newrl

        oldrl = newrl  ## update
        oldseq = newseq
        if (update_times > 1):
            oldrl = autoencoder.predict(onehot_singleseq(oldseq))[1]
            oldrl = scaler.inverse_transform(oldrl).reshape(-1)[0]

    return recorded_seqs

def optimize_sequences(model, scaler, init_seqs, iterations, coef):
    """
    Optimize the input sequences using the autoencoder model.

    Args:
        model: The pre-trained autoencoder model.
        scaler: The scaler used for scaling the output values.
        init_seqs: A list of input sequences to be optimized.
        iterations: The number of iterations for the optimization.
        coef: A coefficient used to control the direction of MRL change during the optimization process.

    Returns:
        ae_seqs: A dictionary containing the optimized sequences and their output values.
    """
    # Split the autoencoder into encoder and decoder
    encoder, decoder, decoded_input_len = split_autoencoder(model)

    nbr_sequences = len(init_seqs)
    ae_seqs = {}
    print("%%% generating new UTR seqs by autoencoder ... %%% ")
    for x in range(nbr_sequences):
        print("Processing optimization for seq No.", x)
        ae_seqs[x] = iter_ae(model, scaler, encoder, decoder, init_seqs[x], coef, decoded_input_len, iterations)

    return ae_seqs