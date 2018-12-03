# -*- coding: UTF-8 -*-
import numpy as np
import setting


def encode(text):
    vector = np.zeros(setting.ALL_CHAR_SET_LEN * setting.MAX_CAPTCHA, dtype=float)

    def char2pos(c):
        if(ord(c)>=65 and ord(c)<=90):
            k=ord(c)-39
        else:
            k=ord(c)-97
        return k
    for i, c in enumerate(text):
        # print(i,':',c)
        idx = i * setting.ALL_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1.0

    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/52
        char_idx = c % setting.ALL_CHAR_SET_LEN
        if char_idx < 26:
            char_code = char_idx + ord('a')
        elif char_idx <52:
            char_code = char_idx + ord('A') - 26
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

if __name__ == '__main__':
    e = encode('AZza')
    print(e)
    print(decode(e))