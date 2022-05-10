import math
import zlib # decompressing if needed
import struct  # parsing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from urllib.parse import unquote #url decode
import base64 # decode base 64
from xml.etree import ElementTree as ET # string to xml file
import xml.dom.minidom as md # beutify xml
import os


'''
    tornado.png -> with PLTE chunk
    dice.png -> with tIME, gAMA
    diagram2.png, diag1.drawio.png -> with tEXt (XML)
'''
file_name = 'diagram2.png'
# file_name = 'Screen_color_test_VGA_256colors.png'
# file_name = 'Screen_color_test_VGA_4colors.png'
file = open(file_name, 'rb')

# image showing
img_show = Image.open(file_name)
img_show.show()


# Checking if it is a valid PNG file that we loaded
PNG_sig = b'\x89PNG\r\n\x1a\n'
if file.read(len(PNG_sig)) != PNG_sig:
    print("Not valid PNG file")

"""
Chunks' format:
1. (4b) Length of 2. and 3. (can be 0)
2. (4b) Type
3. Data (Length-len(Type)
4. (4b) CRC sum
"""


# Function for chunks reading
def chunk_reading(file):
    # getting length and type
    # >I4s is big endian int32(4) and 4*char(4)
    len_of_chunk, type_of_chunk = struct.unpack('>I4s', file.read(8))
    # getting data
    data_of_chunk = file.read(len_of_chunk)
    # calculating crc based on data
    checksum = zlib.crc32(data_of_chunk, zlib.crc32(struct.pack('>4s', type_of_chunk)))
    crc_of_chunk, = struct.unpack('>I', file.read(4))
    if crc_of_chunk != checksum:
        print(f'Calculated CRC: {checksum} is not equal to the one in the file: {crc_of_chunk}')
    return type_of_chunk, data_of_chunk, len_of_chunk, crc_of_chunk


list_of_chunks = []
PLTE_present = False
tEXt_present = False
tEXt_data = []
index = 0
tIME_present = False
gAMA_present = False


while 1:
    ch_type, data, len, crc = (chunk_reading(file))
    list_of_chunks.append((ch_type, data, len, crc))
    if ch_type == b'PLTE':
        PLTE_present = True
    elif ch_type == b'tEXt':
        tEXt_present = True
        tEXt_data.insert(0,data)
        index += 1
    if ch_type == b'tIME':
        tIME_present = True
    elif ch_type == b'gAMA':
        gAMA_present = True
    elif ch_type == b'IEND':
        index -= 1
        break

list_of_chunk_types = []

# chunks showing:
for ch_type, data, len, crc in list_of_chunks:
    print(ch_type, data)
    list_of_chunk_types.append(ch_type)

######################
####### IHDR #########
# IHDR is always first
IHDR_data = list_of_chunks[0][1]

# IHDR is always 13bytes long, contains these objects:
width, height, bit_depth, color_type, compression_m,\
filter_m, interlace_m = struct.unpack('>IIBBBBB', IHDR_data)

# Printing values:
print(f'\nIHDR CHUNK metadata:\nwidth: {width}, height: {height},\nbit_depth: {bit_depth}, color type: {color_type},\n'
      f'compression method: {compression_m}, filter method: {filter_m}, interlace method: {interlace_m}')

##################################
from itertools import zip_longest


def sqrt_int(X: int):
    N = math.floor(math.sqrt(X))
    while bool(X % N):
        N -= 1
    M = X // N
    return M, N


if PLTE_present:
    PLTE_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'PLTE')
    for chunk_type, chunk_data, len, crc in list_of_chunks:
        if chunk_type == b'PLTE' and len % 3 == 0:
            PLTE_len = len

    print(" \ndane do palety barw: ", PLTE_data)
    print("Dlugosc bloku PLTE: "+ str(PLTE_len))
    pixels = struct.unpack('>'+str(PLTE_len)+'B', PLTE_data)
    print(pixels[:12]) # gdy niepogrupowane w trójki

    m, n = sqrt_int(PLTE_len/3)
    pixels = np.reshape(pixels, (int(m), int(n), 3))
    print(pixels.shape)
    fig_1 = plt.figure(1)
    plt.imshow(pixels)
    plt.title('PLTE palette')

########### Optional chunks ##########

##### tEXt chunk ##########

def xml_decode(x_text: str):
    x_text = unquote(x_text)  # url decode
    begin_xml = 0
    end_xml = 0
    char_index = 0
    while end_xml == 0:
        if x_text[char_index - 1] == '>' and x_text[char_index] != '<':
            begin_xml = char_index
        if begin_xml != 0 and x_text[char_index] == '<':
            end_xml = char_index
        char_index += 1
    x_text = x_text[begin_xml:end_xml]
    x_text = base64.b64decode(x_text)  # decode base64
    x_text = zlib.decompress(x_text, -15)  # decompress base 64 string
    x_text = x_text.decode('utf-8', 'replace')  # decode utf-8
    x_text = unquote(x_text)  # url decode
    index_xml = 0
    dom = md.parseString(x_text)
    x_text = dom.toprettyxml()
    x_text = os.linesep.join([s for s in x_text.splitlines()
                            if s.strip()])
    return x_text


if tEXt_present:
    while index >= 0:
        try:
            key, text = tEXt_data[index].split(b'\x00', 1)
            key = key.decode('utf-8', 'replace')
            text = text.decode('utf-8', 'replace')
            if key == 'mxfile':
                text = xml_decode(text)
                xml_text = ET.XML(text)  # umożliwia wpisanie do pliku xml
                with open(file_name[:-4]+".xml", "wb") as f:
                    f.write(ET.tostring(xml_text))
        except ValueError:
            key = None
            value = tEXt_data.decode('utf-8', 'replace')
        print(f'\n(*) tEXt info, {key}: {text}')
        index -= 1


##### tIME chunk ##########

if tIME_present:
    tIME_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'tIME')
    year, mon, day, h, min, sec = struct.unpack('>h5B', tIME_data)

    print(f'\n(*) tIME info, last modification: {day}-{mon}-{year}, {h}:{min}:{sec}')

##### gAMA chunk ##########

if gAMA_present:
    gAMA_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'gAMA')
    gamma, = struct.unpack('>I', gAMA_data)

    print(f'\n(*) gAMA info: Gamma = {gamma/100000}')

print("\nChunk types:", list_of_chunk_types)

# Anonymization of PNG
main_chunks = [b'IHDR', b'IDAT', b'IEND']
if PLTE_present:
    main_chunks.insert(1, b'PLTE')
print(main_chunks)

new_file_name = file_name[:-4] + "_anon.png"
new_file_handler = open(new_file_name, 'wb')
new_file_handler.write(PNG_sig)

for chunk in list_of_chunks:
    if chunk[0] in main_chunks:
        new_file_handler.write(struct.pack('>I', chunk[2]))
        new_file_handler.write(chunk[0])
        new_file_handler.write(chunk[1])
        new_file_handler.write(struct.pack('>I', chunk[3]))

new_file_handler.close()


# FFT function
def fft(image):
    img = cv2.imread(image, 0)

    fourier = np.fft.fft2(img)
    # shifting fft to be centred (składowa stała na środku a nie w rogu)
    fourier_shifted = np.fft.fftshift(fourier)
    # składowa stała >>> od pozostałych f modułu, więc zmiana na skalę log
    fourier_mag = np.asarray(20 * np.log10(np.abs(fourier_shifted)), dtype=np.uint8)
    # aby uzysać fazę użyjemy fcji angle z numpy, która zwraca wartości kątów z fft,
    # czyli fazę z liczby urojonej
    fourier_phase = np.asarray(np.angle(fourier_shifted), dtype=np.uint8)

    fig_2 = plt.figure(2)  # show source image and FFT
    plt.subplot(221), plt.imshow(img, cmap="Greys")
    plt.title('Input Image')

    plt.subplot(222), plt.imshow(fourier_mag, cmap="Greys_r")
    plt.title('FFT Magnitude')

    plt.subplot(223), plt.imshow(fourier_phase, cmap="Greys_r")
    plt.title('FFT Phase')

    # checking if FFT is correct
    fig_3 = plt.figure(3)
    fourier_inverted = np.fft.ifft2(fourier)  # inverting fft

    plt.subplot(121), plt.imshow(img, cmap="Greys")
    plt.title('Input Image')
    plt.subplot(122), plt.imshow(np.asarray(fourier_inverted, dtype=np.uint8), cmap="Greys")
    plt.title('Inverted Image')

    plt.show()


fft(file_name)
