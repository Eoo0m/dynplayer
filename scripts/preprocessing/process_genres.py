"""
Process genre information from spotify_genre_info_sampled.csv
- Keep only the first genre for each track
- Count genres that appear 50 or more times
"""

import csv
from collections import Counter
from pathlib import Path

def process_genres():
    input_file = Path("linear_evaluation/spotify_genre_info_sampled.csv")
    output_file = Path("linear_evaluation/spotify_genre_info_sampled_first_genre.csv")

    genre_counter = Counter()
    rows_to_write = []

    # Read and process the CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            genres = row['genres'].strip()

            # Extract first genre if genres exist
            if genres:
                # Split by comma and take the first one
                first_genre = genres.split(',')[0].strip()
                genre_counter[first_genre] += 1
                row['genres'] = first_genre
            else:
                # Keep empty as is
                row['genres'] = ''

            rows_to_write.append(row)

    # Write the modified CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"✅ Wrote modified CSV to {output_file}")
    print(f"\nTotal tracks processed: {len(rows_to_write):,}")
    print(f"Total unique genres: {len(genre_counter):,}")

    # Find genres with 50+ occurrences
    frequent_genres = {genre: count for genre, count in genre_counter.items()
                       if count >= 50}

    print(f"\n=== Genres with 50+ occurrences: {len(frequent_genres)} ===")

    # Sort by count (descending)
    sorted_genres = sorted(frequent_genres.items(), key=lambda x: x[1], reverse=True)

    for genre, count in sorted_genres:
        print(f"{genre:30s} : {count:,}")

    print(f"\nTotal genres with ≥50 occurrences: {len(frequent_genres)}")

    return frequent_genres

if __name__ == "__main__":
    process_genres()
