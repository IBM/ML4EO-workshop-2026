# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0

import os
import re
import csv

from datetime import datetime, timedelta

def extract_dates(filename):
    """Extract date from filename in format saltmarsh_category_{Month}{Year}_...
    For single months, uses day 15. For multi-month ranges, calculates midpoint date.
    """
    
    # Month name to number mapping
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Match pattern: MonthName(s)Year (e.g., April2019, July-August2016)
    match = re.search(r'([A-Z][a-z]+(?:-[A-Z][a-z]+)*?)(\d{4})', filename)
    if match:
        month_str = match.group(1)
        year = int(match.group(2))
        
        # Split months if hyphenated
        months = month_str.split('-')
        
        if len(months) == 1:
            # Single month: use 15th
            month_num = month_map.get(months[0])
            if month_num:
                return f"{year}-{month_num:02d}-15"
        else:
            # Multiple months: calculate midpoint
            first_month = month_map.get(months[0])
            last_month = month_map.get(months[-1])
            
            if first_month and last_month:
                # Create start date (1st of first month) and end date (last day of last month)
                start_date = datetime(year, first_month, 1)
                
                # Get last day of last month
                if last_month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, last_month + 1, 1) - timedelta(days=1)
                
                # Calculate midpoint
                midpoint = start_date + (end_date - start_date) / 2
                return midpoint.strftime("%Y-%m-%d")
    
    return None

def marsh_labels_to_csv(labels_dir):
    """Write labels to csv file."""
    
    # labels_dir = "/Users/rosielickorish/Library/CloudStorage/Box-Box/hncdi-saltmarsh/data/processed/categorical/rgb_rgbn_spring_to_autumn_dates_multi-campaign/res_10m/test/labels_extent"
    output_csv = f"{labels_dir}/metadata.csv"
    
    # Get all files from directory
    files = sorted(os.listdir(labels_dir))
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'date'])
        
        for filename in files:
            if os.path.isfile(os.path.join(labels_dir, filename)):
                date = extract_dates(filename)
                if date:
                    writer.writerow([filename, date])

    print(f"Created {output_csv}")
    print("testing")

    with open(output_csv) as input_file:
        for i, line in enumerate(input_file):
            if i >= 5:
                print("...")
                break
            print(line.rstrip())

