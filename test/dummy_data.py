import os
import csv
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial


def generate_dummy_audio(duration, sample_rate=22050):
    """Generate dummy audio data."""
    return np.random.randn(int(duration * sample_rate))


def create_single_sample(args):
    """Create a single dummy sample."""
    i, audio_path, min_duration, max_duration, speakers = args

    # Generate dummy audio
    duration = np.random.uniform(min_duration, max_duration)
    audio = generate_dummy_audio(duration)

    # Create dummy metadata
    speaker_id = np.random.choice(speakers)
    file_name = f'audio_{i:04d}'
    text = f"This is a dummy text for audio sample {i}."

    # Save audio file
    audio_file = os.path.join(audio_path, f'{file_name}.wav')
    sf.write(audio_file, audio, 22050)

    return [file_name, text, speaker_id]


def create_dummy_dataset(base_path, num_samples=100, min_duration=5, max_duration=10, parallel=False, num_workers=None):
    """Create a dummy dataset for TTS preprocessing testing."""
    dummy_dataset_path = os.path.join(base_path, 'dummy_dataset')
    audio_path = os.path.join(dummy_dataset_path, 'audio')
    os.makedirs(audio_path, exist_ok=True)

    speakers = [f'speaker_{i:03d}' for i in range(5)]  # 5 dummy speakers

    if parallel:
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        with multiprocessing.Pool(num_workers) as pool:
            args_list = [(i, audio_path, min_duration, max_duration, speakers) for i in range(num_samples)]
            metadata = list(tqdm(pool.imap(create_single_sample, args_list), total=num_samples, desc="Generating dummy data"))
    else:
        metadata = []
        for i in tqdm(range(num_samples), desc="Generating dummy data"):
            metadata.append(create_single_sample((i, audio_path, min_duration, max_duration, speakers)))

    metadata_file = os.path.join(dummy_dataset_path, 'metadata.csv')
    with open(metadata_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['file_name', 'text', 'speaker_id'])
        writer.writerows(metadata)

    print(f"Dummy dataset created at {base_path}")
    print(f"Total samples: {num_samples}")
    print(f"Metadata file: {metadata_file}")
    return dummy_dataset_path


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    base_path = "E:\\ML\\Datasets"
    parallel = True  # Set to False to disable parallel processing
    num_workers = None  # Adjust this based on your system's capabilities
    num_samples = 10000

    create_dummy_dataset(base_path, num_samples=num_samples, parallel=parallel, num_workers=num_workers)

    # Display a few rows of the generated metadata
    metadata_df = pd.read_csv(os.path.join(os.path.join(base_path, 'dummy_dataset'), 'metadata.csv'), sep='|')
    print("\nSample of generated metadata:")
    print(metadata_df.head())

    # Display the directory structure
    print("\nGenerated directory structure:")
    for root, dirs, files in os.walk(os.path.join(base_path, 'dummy_dataset')):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))