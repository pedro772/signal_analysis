import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import librosa as lr

# carrega o arquivo de áudio com a taxa de amostragem de 22,05 kHz
signal, samplerate = lr.load('beat_one.wav', sr=22050)

# número de janelas de tempo (time_segments) = duração do arquivo / duração de cada janela (2048/22050) * 4 (sobreposição de 75%)

# método spectrogram de scipy.signal retorna o spectrograma como o quadrado da magnitude da stft
# argumento 'magnitude' convertendo para um valor absoluto e 'spectrum' para gerar o spectrograma com unidade V²
frequencies, time_segments, spectrogram = sc.signal.spectrogram(signal, fs=samplerate, window='hann', nperseg=1025, noverlap=512, nfft=2048, scaling='spectrum', mode='magnitude')

print(spectrogram.mean()) # Resposta da questão 1

# converte a matrix obtida anteriormente para dBs (com potencial de referência = 1) usando a o método power_to_db da biblioteca librosa
spectrogram_dbs = lr.power_to_db(spectrogram, ref=1)

print(spectrogram_dbs.mean()) # Resposta da questão 2

# cria o vetor da etapa E (ainda vazio)
limit = len(time_segments)
max_powers = np.arange(limit, dtype='float32')

# armazena o índice da frequência (eixo 0 da matriz da etapa C) da maior potência em cada janela de tempo
index_of_max_power = np.argmax(spectrogram, axis=0)

# armazena as frequências (com os índices obtidos anteriormente) no vetor da etapa E
for i in range(limit):
    max_powers[i] = frequencies[index_of_max_power[i]]

print(max_powers.mean()) # Resposta da questão 3

# Ordena vetor da etapa E em ordem crescente
sorted_array = np.sort(max_powers)

# Obtém o percentil 75 do vetor da etapa E
percentile_75 = np.percentile(sorted_array, 75)
print(percentile_75) # Resposta da questão 4

# Obtém os percentis 25, 50 e 100 (preparação para a questão 5)
percentile_25 = np.percentile(sorted_array, 25)
percentile_50 = np.percentile(sorted_array, 50)
percentile_100 = np.percentile(sorted_array, 100)

# Armazena os índice de cada quartil da etapa F
percentile_indexes = [
    np.nonzero(sorted_array == percentile_25)[0][0], # [q0, q1) [0, 953)
    np.nonzero(sorted_array == percentile_50)[0][0], # [q1, q2) [953, 3713) 
    # [q1, q2) tem 2760 elementos do vetor da etapa E
    # assim, é o intervalo que possui mais elementos do mesmo 
    np.nonzero(sorted_array == percentile_75)[0][0], # [q2, q3) [3713, 5731)
    np.nonzero(sorted_array == percentile_100)[0][0] # [q3, q4) [5731, 7734)
]

print((percentile_indexes[1] - percentile_indexes[0]) / len(time_segments)) # Resposta da questão 6

# cria o vetor da etapa G (ainda vazio)
binaries = np.arange(limit - 1, dtype='int32')

# Preenche o vetor da etapa G
for i in range(limit - 1):
    # Os índices obtidos anteriormente classificam as janelas que mudam de intervalo (valor 1 no vetor da etapa G)
    if(i == percentile_indexes[0] or i == percentile_indexes[1] or i == percentile_indexes[2]):
        binaries[i] = 1
    else:
        binaries[i] = 0

# janela equivalente 30 segundos
window_size = int(30 / (2048 / 22050) * 4)

# função para calcular a média móvel simples (SMA em inglês)
def moving_average (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# vetor da etapa H
smas = moving_average(binaries, window_size)

# Índice do valor do pico do vetor da etapa H
peak_value_index = np.argmax(smas)

print(smas[peak_value_index]) # Resposta da questão 6

# Obtém o índice da janela de tempo que contém o pico indicado anteriormente
# Devido ao vetor ser um vetor de floats, é necessário usar o método isclose(), que retorna os valores próximos do indicado, assim, perde-se um pouco da precisão
i, j = np.where(np.isclose(spectrogram, smas[peak_value_index]))

print(time_segments[j[0]]) # Resposta da questão 7

# Gera e mostra a figura com o gráfico representando o vetor da etapa H
plt.plot(smas)
plt.xlabel('Janelas de tempo')
plt.ylabel('Densidade espectral (V²)')
plt.savefig('figura_do_vetor_media_movel')
plt.show()