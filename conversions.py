import math


# Designed By Thaising
# Combined Master-PhD in MSISLAB
def Floating2Binary(num, Exponent_Bit, Mantissa_Bit):
    sign = ('1' if num < 0 else '0')
    num = abs(num)
    bias = (2 ** (Exponent_Bit - 1) - 1)
    e = (0 if num == 0 else math.floor(math.log(num, 2) + bias))
    if e > (2 ** Exponent_Bit - 2):  # overflow
        exponent = '1' * Exponent_Bit
        mantissa = '0' * Mantissa_Bit
    else:
        if e > 0:  # normal
            s = num / 2 ** (e - bias) - 1
            exponent = bin(e)[2:].zfill(Exponent_Bit)
        else:  # submoral
            s = num / 2 ** (-bias + 1)
            exponent = '0' * Exponent_Bit
        mantissa = bin(int(s * (2 ** Mantissa_Bit) + 0.5))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
    return sign + exponent + mantissa



# 2023/04/29: Designed By Thaising
# Combined Master-PhD in MSISLAB
def Binary2Floating(s, Exponent_Bit, Mantissa_Bit):
    neg = int(s[0], 2)
    if int(s[1:1 + Exponent_Bit], 2) != 0:
        exponent = int(s[1:1 + Exponent_Bit], 2) - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(s[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit) + 1
    else:  # subnormal
        exponent = 1 - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(s[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit)
    return ((-1) ** neg) * (2 ** exponent) * mantissa