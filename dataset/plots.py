import re
import matplotlib.pyplot as plt

# Map threads per block to file names
threads_per_block = [128, 256, 512, 1024]

# Example structure: change the paths to your actual txt file paths
files_small = {
    128: 'timing_par_km_128/timing_small.txt',
    256: 'timing_par_km_256/timing_small.txt',
    512: 'timing_par_km_512/timing_small.txt',
    1024: 'timing_par_km_1024/timing_small.txt',
}

files_medium = {
    128: 'timing_par_km_128/timing_medium.txt',
    256: 'timing_par_km_256/timing_medium.txt',
    512: 'timing_par_km_512/timing_medium.txt',
    1024: 'timing_par_km_1024/timing_medium.txt',
}

files_large = {
    128: 'timing_par_km_128/timing_large.txt',
    256: 'timing_par_km_256/timing_large.txt',
    512: 'timing_par_km_512/timing_large.txt',
    1024: 'timing_par_km_1024/timing_large.txt',
}

# Sequential baseline files
sequential_files = {
    'small': 'timing_seq/timing_small.txt',
    'medium': 'timing_seq/timing_medium.txt',
    'large': 'timing_seq/timing_large.txt',
}


def extract_total_time(filepath):
    """Extracts the Total time from a timing file (sum of background, foreground, beta, and weights)"""
    total_background = 0
    total_foreground = 0
    total_beta = 0
    total_weights = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if "Total time for background:" in line:
                total_background = float(re.search(r'([\d.]+)', line).group(1))
            elif "Total time for foreground:" in line:
                total_foreground = float(re.search(r'([\d.]+)', line).group(1))
            elif "Total time for Beta calculation:" in line:
                total_beta = float(re.search(r'([\d.]+)', line).group(1))
            elif "Total time for Weights calculation:" in line:
                total_weights = float(re.search(r'([\d.]+)', line).group(1))

    total = total_background + total_foreground + total_beta + total_weights
    return total

def compute_speedups(files, sequential_total):
    speedups = []
    for threads in threads_per_block:
        total = extract_total_time(files[threads])
        speedup = sequential_total / total
        speedups.append(speedup)
    return speedups

def main():
    # Get baseline sequential times
    seq_small = extract_total_time(sequential_files['small'])
    seq_medium = extract_total_time(sequential_files['medium'])
    seq_large = extract_total_time(sequential_files['large'])

    # Compute speedups
    speedups_small = compute_speedups(files_small, seq_small)
    speedups_medium = compute_speedups(files_medium, seq_medium)
    speedups_large = compute_speedups(files_large, seq_large)

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(threads_per_block, speedups_small, marker='o', label='Small Images')
    plt.plot(threads_per_block, speedups_medium, marker='s', label='Medium Images')
    plt.plot(threads_per_block, speedups_large, marker='^', label='Large Images')

    plt.xlabel('Threads per Block')
    plt.ylabel('Speedup over Sequential')
    plt.title('Speedup vs Threads per Block')
    plt.legend()
    plt.grid(True)
    plt.xticks(threads_per_block)
    plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
