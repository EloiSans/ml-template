import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read CSV and transform to MarkDown or LaTex")
    parser.add_argument("--input-file", type=str, help="Which CSV to read")
    parser.add_argument("--format", type=str, help="Which format transform to", choices=["mkd", "tex"], default='mkd')
    parser.add_argument("--output-file", type=str, help="Where to write the output")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    if args.output_file:
        with open(args.output_file, 'a') as buf:
            if args.format == 'mkd':
                df.to_markdown(buf=buf, mode='a')
            else:
                df.to_latex(buf=buf)
    else:
        if args.format == 'mkd':
            print(df.to_markdown())
        else:
            print(df.to_latex())
