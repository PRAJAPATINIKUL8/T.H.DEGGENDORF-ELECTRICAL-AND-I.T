# Week 02 Viva Preparation

**Student:** NIKUL<br>
**Semester:** WS-26/27<br>
**Course:** CONTACTLESS SENSORS

## Questions and concise answers

1. **What is the Fourier transform used for here?**<br>
   It shows how the signal's amplitude is distributed over frequency.

2. **What does the FFT calculate?**<br>
   The FFT is an efficient algorithm for calculating the discrete Fourier transform of sampled data.

3. **Why does a single rectangle transform into a sinc?**<br>
   The Fourier-transform integral of a constant over a finite interval evaluates to a sine divided by frequency. After normalization, that is `np.sinc(W * f)`.

4. **What does `np.sinc(z)` mean in NumPy?**<br>
   NumPy uses the normalized sinc, `sin(pi*z)/(pi*z)`, with value 1 at zero.

5. **Why is the rectangle's height `1/W`?**<br>
   Its width times its height is `W * (1/W) = 1`, so its continuous area is one.

6. **Why does increasing `W` narrow the transform?**<br>
   The first sinc zero is at `|f| = 1/W`. Larger `W` moves this zero closer to zero frequency.

7. **What happens when the rectangle gets narrower?**<br>
   Its first zero moves farther from zero frequency, so the spectrum becomes broader.

8. **Why multiply the FFT by the sampling interval `dx` for the single pulse?**<br>
   The continuous transform is an integral. Multiplying the sample sum by `dx` approximates that integral and gives the zero-frequency value as the sampled area.

9. **What is the FFT value at zero frequency in this notebook?**<br>
   It is the sum of the rectangle samples times `dx`, which is approximately its area, one.

10. **What does `np.fft.fftfreq` return?**<br>
    It returns the frequency coordinates for the FFT bins, in cycles per unit of the sample spacing. With seconds as the sample unit, the result is in hertz.

11. **What does `fftshift` do?**<br>
    It rearranges the FFT and frequency axis so that zero frequency is in the center, with negative frequencies to the left and positive frequencies to the right.

12. **Why does the notebook use `ifftshift` before the single-pulse FFT?**<br>
    The rectangle samples are arranged around the middle of the array. `ifftshift` moves that centered sample arrangement to the order expected by the FFT.

13. **Why is there a half-sample phase correction for the single pulse?**<br>
    The time samples are centered between grid points. The phase factor accounts for that half-step so the numerical transform is referenced to time zero.

14. **What is frequency resolution?**<br>
    It is the spacing between neighboring FFT frequencies, `1/T`, where `T` is the record duration. A longer record gives smaller frequency spacing.

15. **Why choose an integer number of periods for the train?**<br>
    The sampled record then joins smoothly to its periodic FFT repetition, and the harmonics fall exactly on FFT bins. A partial period can cause spectral leakage.

16. **Why does a periodic rectangle train have discrete harmonic peaks?**<br>
    A repeating signal can be described by frequencies at the repetition frequency and its integer multiples. These frequencies are its harmonics.

17. **Why are some harmonics missing for a 50% duty-cycle train?**<br>
    The centered 50% rectangle has zero coefficients at the even nonzero harmonic orders. Its remaining harmonic amplitudes follow a sinc-shaped envelope.

18. **Does the professor's cosine-sign formula let us adjust `W`?**<br>
    No. It fixes the duty cycle at 50%. The notebook's width experiment uses a separate modulo-based rectangle definition, where `W` is the pulse width and the period remains fixed.

19. **What changes when pulse width or duty cycle changes?**<br>
    The DC coefficient equals the duty cycle, and the harmonic envelope changes. Wider pulses have a narrower envelope in harmonic order; narrower pulses have a broader envelope.

20. **What is the difference between a single rectangle and a periodic train?**<br>
    A single isolated rectangle has a continuous sinc-shaped transform. A periodic train has discrete harmonic lines whose amplitudes follow a sinc-like envelope.
