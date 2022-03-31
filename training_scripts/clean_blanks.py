import csv
import os


def dedup_label(filename):
    dedup = {}
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if dedup.get(int(row[0]), "b") == "b":
                dedup[int(row[0])] = row[1]
            else:
                print(f"skipped frame {int(row[0])} in file {filename}")
                continue
    return dedup


if __name__ == "__main__":

    train_dirs = [
        "data/bengals-ravens",
        "data/browns-ravens",
        "data/bears-ravens",
        "data/dolphins-ravens",
        "data/ravens-browns",
        "data/ravens-bengals",
        "data/ravens-packers",
        "data/steelers-ravens",
    ]

    train_locs = [f"{d}/frames.csv" for d in train_dirs]

    for f in train_locs:
        print(f"Processing {f}")
        deduped = dedup_label(f)
        with open(f"{os.path.splitext(f)[0]}_clean.csv", "w", newline="") as f:
            cw = csv.writer(f, delimiter=",")
            for k, v in deduped.items():
                cw.writerow((k, v))
