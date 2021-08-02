import json
from argparse import ArgumentParser
from utils.dataset import get_train_val_test_split
import os


def main(args):

    train_list, val_list, test_list, label_map = get_train_val_test_split(args.data_root, args.val_list_file, args.test_list_file)

    with open(os.path.join(args.out_dir, "training_list.txt"), "w+") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(args.out_dir, "validation_list.txt"), "w+") as f:
        f.write("\n".join(val_list))

    with open(os.path.join(args.out_dir, "testing_list.txt"), "w+") as f:
        f.write("\n".join(test_list))

    with open(os.path.join(args.out_dir, "label_map.json"), "w+") as f:
        json.dump(label_map, f)

    print("Saved data lists and label map.")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--val_list_file", type=str, required=True, help="Path to validation_list.txt.")
    parser.add_argument("-t", "--test_list_file", type=str, required=True, help="Path to test_list.txt.")
    parser.add_argument("-d", "--data_root", type=str, required=True, help="Root directory of speech commands v2 dataset.")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory for data lists and label map.")
    args = parser.parse_args()

    main(args)