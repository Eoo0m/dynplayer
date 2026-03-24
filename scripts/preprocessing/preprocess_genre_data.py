"""
Preprocess genre data from /data/spotify_genre_info_sampled.csv
- Keep only the first genre for each track
- Filter to keep only tracks with frequent genres (≥50 occurrences)
"""

import csv
from collections import Counter
from pathlib import Path

def preprocess_genre_data():
    input_file = Path("data/spotify_genre_info_sampled.csv")
    output_file = Path("data/spotify_genre_info_frequent.csv")

    # First pass: count first genres
    genre_counter = Counter()
    all_rows = []

    print(f"Reading {input_file}...")
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

    print(f"\nTotal tracks: {len(all_rows):,}")
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

    print(f"Total genres: {len(sorted_genres)}")
    print(f"\nTop 20 genres:")
    for genre, count in sorted_genres[:20]:
        print(f"{genre:30s} : {count:,}")

    # Save genre list
    genre_list_file = Path("data/genre_list.txt")
    with open(genre_list_file, 'w', encoding='utf-8') as f:
        for genre, _ in sorted_genres:
            f.write(f"{genre}\n")

    print(f"\n✅ Saved genre list to {genre_list_file}")

    return filtered_rows, sorted_genres

if __name__ == "__main__":
    preprocess_genre_data()
