"""
Filter spotify_genre_info_sampled.csv to keep only tracks with frequent genres (≥50 occurrences)
"""

import csv
from collections import Counter
from pathlib import Path

def filter_frequent_genres():
    input_file = Path("linear_evaluation/spotify_genre_info_sampled.csv")
    output_file = Path("linear_evaluation/spotify_genre_info_frequent.csv")

    # First pass: count first genres
    genre_counter = Counter()
    all_rows = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            genres = row['genres'].strip()

            if genres:
                # Extract first genre
                first_genre = genres.split(',')[0].strip()
                genre_counter[first_genre] += 1
            else:
                first_genre = ''

            all_rows.append((row, first_genre))

    # Find frequent genres (≥50 occurrences)
    frequent_genres = {genre for genre, count in genre_counter.items() if count >= 50}

    print(f"Total tracks: {len(all_rows):,}")
    print(f"Frequent genres (≥50): {len(frequent_genres)}")

    # Second pass: filter rows with frequent genres
    filtered_rows = []
    for row, first_genre in all_rows:
        if first_genre in frequent_genres:
            # Update the row to contain only the first genre
            row['genres'] = first_genre
            filtered_rows.append(row)

    # Write filtered CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Filtered tracks: {len(filtered_rows):,}")
    print(f"✅ Wrote filtered CSV to {output_file}")

    # Show genre distribution
    filtered_genre_counter = Counter()
    for row in filtered_rows:
        filtered_genre_counter[row['genres']] += 1

    print(f"\n=== Genre distribution in filtered dataset ===")
    sorted_genres = sorted(filtered_genre_counter.items(), key=lambda x: x[1], reverse=True)

    for genre, count in sorted_genres[:20]:  # Show top 20
        print(f"{genre:30s} : {count:,}")

    if len(sorted_genres) > 20:
        print(f"... and {len(sorted_genres) - 20} more genres")

if __name__ == "__main__":
    filter_frequent_genres()
