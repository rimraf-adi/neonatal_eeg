Input epoch        : (B, 18, 256)

Wavelet D1–D4      : (B, 18, 128/64/32/16)

CNN features       : (B, 64, T_i)

Temporal attention : (B, 64) × 4

Concatenation      : (B, 256)

Dense classifier   : (B, 1)
