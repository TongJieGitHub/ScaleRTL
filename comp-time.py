from pathlib import Path
import re
import os
import pandas as pd

# Read benchmark names from case_list.txt
case_list_file = Path('case_list.txt')
if not case_list_file.exists():
    raise FileNotFoundError(f"Benchmark list file '{case_list_file}' not found.")

with case_list_file.open() as f:
    benchmarks = [line.strip() for line in f if line.strip()]

# Read simulator names from the SIMS environment variable
sims_env = os.environ.get('SIMS', '').strip()
if not sims_env:
    raise ValueError("Environment variable 'SIMS' is not set or is empty.")
sims = re.split(r'\s+', sims_env)

df = []

# Regex to extract "design" and number from benchmark name
pattern = re.compile(r'(\w+)_extended_(\d+)')
print(sims)
for bench in benchmarks:
    for sim in sims:
        log_path = Path('runs', sim, 'obj', bench, 'compile.log')
        if not log_path.exists():
            continue  # Skip if log file doesn't exist
        
        data = log_path.read_text()

        # Find all occurrences of "Wall Time: <number>"
        times = re.findall(r'Wall Time: ([\d.]+)', data)
        
        if times:
            # Sum up all extracted times
            total_time = sum(float(t) for t in times)
            
            # Convert milliseconds to seconds
            total_time_sec = total_time / 1000  

            # Extract design name and number
            match = pattern.match(bench)
            if match:
                design_name = match.group(1)  # Extract design name (e.g., Conv2D)
                design_number = int(match.group(2))  # Extract number (e.g., 1, 2, 4)
            else:
                design_name = bench  # If pattern doesn't match, keep the full name
                design_number = None  # No number extracted
            
            df.append({
                'benchmark': bench,
                'design': design_name,
                'design_number': design_number,
                'simulator': sim,
                'comp_time': total_time,      # Original time in milliseconds
                'comp_time_sec': total_time_sec  # Converted time in seconds
            })
        else:
            print(f"Warning: No Wall Time found in {log_path}")

# Create output directory if not exists
output_dir = Path('comp-time')
output_dir.mkdir(exist_ok=True)

# Save to CSV
output_file = output_dir / 'comp-time.csv'
pd.DataFrame(df).to_csv(output_file, index=False)

print(f"CSV saved to {output_file}")
