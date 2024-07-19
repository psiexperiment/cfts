# CFTS

## Introduction

Cochlear Function Test Suite (CFTS) is a peripheral auditory testing framework
built on top of psiexperiment that includes the following experiments:

* Auditory brainstem response (ABR)
  * Conventional ABR
  * Interleaved ABR (https://doi.org/10.1007/s10162-020-00754-3)
* Distortion product otoacoustic emission (DPOAE)
  * Input-output functions (monaural, binaural)
  * DP-grams (monaural)
* Envelope following response (EFR)
  * Sinusoidal amplitude modulation (SAM)
  * Rectangular amplitude modulation (RAM)
* Middle ear muscle reflex (MEMR), also known as the wideband acoustic reflex
  * Keefe (ipsilateral, contralateral)
  * Valero (contralateral)
  * Sweep (contralateral)

CFTS nominally supports both National Instruments and TDT hardware. Those using National Instruments hardware will have to do some setup work to map the appropriate inputs and outputs as well as ensure that all devices are running off of the same sample clock (to ensure precise timing).

For TDT hardware, only the RZ6 is currently supported; however, the RZ6 is standard so no customization should be needed to run CFTS on top of the RZ6.

## Installing

Install your preferred Python distribution. For use with National Instruments hardware:

    pip install cfts[ni]

For use with TDT hardware:

    pip install cfts[tdt]
