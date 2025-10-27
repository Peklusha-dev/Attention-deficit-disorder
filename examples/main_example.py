"""
Main Example: Complete Eye Movement Tracking Pipeline

This script demonstrates how to:
1. Detect pupils from video/images
2. Segment eye movements using different algorithms
3. Visualize the results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import PupilDetector
from src.segmentation import IVTSegmenter, IDTSegmenter, HMMSegmenter
from src.visualization import EyeMovementPlotter


def main():
    """Main example pipeline."""
    
    # Configuration
    video_path = 'data/sample_video.mp4'  # Update with your video path
    pupil_csv = 'pupil_data.csv'
    
    print("=== Eye Movement Tracking System ===\n")
    
    # Step 1: Detect pupils
    print("Step 1: Detecting pupils from video...")
    detector = PupilDetector()
    detector.process_video(video_path, output_csv=pupil_csv, show_result=False)
    print("✓ Pupil detection complete\n")
    
    # Step 2: Segment eye movements (I-VT)
    print("Step 2: Segmenting eye movements using I-VT algorithm...")
    ivt_segmenter = IVTSegmenter(velocity_threshold=110, min_fixation_duration=0.1)
    ivt_result = ivt_segmenter.segment(pupil_csv)
    
    ivt_result['fixations'].to_csv('fixations_ivt.csv', index=False)
    ivt_result['saccades'].to_csv('saccades_ivt.csv', index=False)
    print("✓ I-VT segmentation complete\n")
    
    # Step 3: Segment using I-DT
    print("Step 3: Segmenting using I-DT algorithm...")
    idt_segmenter = IDTSegmenter(dispersion_threshold=7, min_fixation_duration=0.1)
    idt_result = idt_segmenter.segment(pupil_csv)
    
    idt_result['fixations'].to_csv('fixations_idt.csv', index=False)
    idt_result['saccades'].to_csv('saccades_idt.csv', index=False)
    print("✓ I-DT segmentation complete\n")
    
    # Step 4: Visualize results
    print("Step 4: Generating visualizations...")
    plotter = EyeMovementPlotter()
    
    plotter.plot_raw_data(pupil_csv, 'plots/raw_data.html')
    plotter.plot_with_segmentation(
        pupil_csv,
        'fixations_ivt.csv',
        'saccades_ivt.csv',
        'plots/segmented_ivt.html'
    )
    print("✓ Visualization complete\n")
    
    print("Done! Check the 'plots' folder for HTML files.")


if __name__ == "__main__":
    main()

