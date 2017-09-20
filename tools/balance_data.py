import argparse
import json
import numpy as np
import os
from random import shuffle


def main(args):

    if args.dataset == 'hotel_reviews':
        dataset_path = os.path.join(args.path, args.dataset, 'train', 'train.txt')
        with open(dataset_path, 'r') as f:
            text = f.read().splitlines()

        ratings = np.array([int(json.loads(item)['ratings']['overall']) for item in text])
        min_samples = np.min(np.unique(ratings, return_counts=True)[1][1:])

        for i in range(args.num_outputs):
            with open(os.path.join(args.path, args.dataset, 'train','output_file_{}.txt'.format(i)),'w') as f:
                data_list = []
                for j in [1,2,3,4,5]:
                    locs = np.where(ratings == j)[0][:min_samples]
                    np.random.shuffle(locs)
                    for loc in locs:
                        data_list.append(text[loc])
                shuffle(data_list)
                _ = [f.write(item + '\n') for item in data_list]
    elif args.dataset == 'amazon_reviews_de':
        dataset_path = os.path.join(args.path, args.dataset, 'train', 'train.txt')
        with open(dataset_path, 'r') as f:
            text = f.read().splitlines()

        # ratings = np.array([int(json.loads(item)['review_rating']) for item in text])
        ratings = []
        for item in text:
            try:
                ratings.append(int(json.loads(item)['review_rating']))
            except:
                pass
        ratings = np.array(ratings)
        min_samples = np.min(np.unique(ratings, return_counts=True)[1][1:])

        for i in range(args.num_outputs):
            with open(os.path.join(args.path, args.dataset, 'train','output_file_{}.txt'.format(i)),'w') as f:
                data_list = []
                for j in [1,2,3,4,5]:
                    locs = np.where(ratings == j)[0][:min_samples]
                    np.random.shuffle(locs)
                    for loc in locs:
                        data_list.append(text[loc])
                shuffle(data_list)
                _ = [f.write(item +'\n') for item in data_list]
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset',
                        help='Which dataset to balance. (Possible values: amazon_reviews_de, hotel_reviews')
    parser.add_argument('--path', help='Path to the dataset.', default='/scratch/OSA/data/datasets/')
    parser.add_argument('--num-outputs', type=int, help='Number of data-balanced output files to generate.', default=10)

    args = parser.parse_args()

    if args.dataset not in ['amazon_reviews_de', 'hotel_reviews']:
        raise NotImplementedError('Dataset {} has not been '
                                  'implemented yet'.format(args.dataset))

    main(args)
