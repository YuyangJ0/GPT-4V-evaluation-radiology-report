import argparse
import pandas as pd
from CXRMetric.run_eval import calc_metric
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, 
                        help="Path to CSV file of groundtruth reports.")
    parser.add_argument("--gen_path", type=str, 
                        help="Path to CSV file of generated reports.")
    parser.add_argument("--out_path", type=str, default="gen_metrics.csv",
                        help="Path to CSV file containing performance scores.")

    args = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args, _ = parse_args()
    df_gt = pd.read_csv(args.gt_path).sort_values(by="study_id")
    df_gen = pd.read_csv(args.gen_path).sort_values(by="study_id")
    df_gt['study_id'] = df_gt['study_id'].apply(str)
    df_gen['study_id'] = df_gen['study_id'].apply(str)

    # deal with null values (findings might exist null values)
    df_gen.dropna(inplace=True)
    df_gt.dropna(inplace=True)
    df_gt = df_gt[df_gt['study_id'].isin(set(df_gen['study_id']))]
    df_gen = df_gen[df_gen['study_id'].isin(set(df_gt['study_id']))]

    assert df_gt.shape[0] == df_gen.shape[0], "The number of rows in df_gt and df_gen are not equal."

    pth_gt = os.path.dirname(args.gt_path)
    pth_gen = os.path.dirname(args.gen_path)
    gen_name = args.gen_path.split('/')[-1].split('.')[0]

    pth_gt_new = os.path.join(pth_gt, 'gt_imp_sample_post.csv')
    pth_gen_new = os.path.join(pth_gen, gen_name+'_post.csv')
    df_gt.to_csv(pth_gt_new, index=False)
    df_gen.to_csv(pth_gen_new, index=False)

    calc_metric(pth_gt_new, pth_gen_new, args.out_path, use_idf=False)