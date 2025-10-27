"""
Eye Movement Segmentation Module

Provides algorithms for segmenting eye movements into fixations and saccades:
- IVT (I-VT): Velocity-based thresholding
- IDT (I-DT): Dispersion-based algorithm
- HMM: Hidden Markov Model-based classification
"""

from .ivt import IVTSegmenter
from .idt import IDTSegmenter
from .hmm import HMMSegmenter

__all__ = ['IVTSegmenter', 'IDTSegmenter', 'HMMSegmenter']

