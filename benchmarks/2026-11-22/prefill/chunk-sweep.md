# Cold Pi prefill with larger scheduler chunks

The test used the EXL3 adaptive DSpark text profile, one B70 at 180 W, and the same Pi prompts across five engine starts. Each start processed two uncached requests at each length. Rates below are medians of the individual engine rates. First-output times include client and server overhead.

| Input tokens | 4,096 chunk (tok/s) | 6,144 chunk (tok/s) | 8,192 chunk (tok/s) | First output, 4,096 → 6,144 |
| ---: | ---: | ---: | ---: | ---: |
| 4,844 | 1,871 | 1,926 | 1,925 | 2.66 → 2.58 s |
| 16,635 | 1,630 | 1,819 | 1,600 | 10.30 → 9.23 s |
| 31,376 | 1,773 | 1,788 | 1,791 | 17.77 → 17.62 s |

The 6,144-token chunk gained 11.6% at 16,635 input tokens and 0.9% at 31,376. Each 4,096 and 6,144 setting had two independent starts. The 8,192 setting had one start. At 31,376 input tokens the lowest sampled free VRAM was 5.37, 4.07 and 2.81 GiB respectively (whole-card DRM Xe readings once per second, sampling available from the second arm).

All five starts ran the same seeded routing repair through Pi with an 8,192-token thinking budget and accepted DSpark proposals. All five patches passed ten hidden tests. Both 4,096 runs returned within the four-minute deadline (209 and 142 s). One 6,144 run returned in 163 s and the other reached the deadline with a passing patch on disk. The 8,192 run returned in 184 s. These timings give no consistent advantage in end-to-end repair latency for the 6,144 setting. The maximum request measured at 6,144 was 31,376 input tokens. [Per-request counters, grades, memory and hashes](chunk-sweep.json)
