import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="data/seg_sentence_3.txt")
parser.add_argument("--learning-rate", type=float, default=0.0004)
parser.add_argument("--work-type", type=str, default="train")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--embed-dim", type=int, default=300)
parser.add_argument("--hidden-dim", type=int, default=600)
parser.add_argument("--fix-length", type=int, default=100)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden-dim-fc1", type=int, default=300)
parser.add_argument("--hidden-dim-fc2", type=int, default=50)
parser.add_argument("--weight-decay", type=float, default=0.000001)
parser.add_argument("--epoch", type=int, default=10)
args = parser.parse_args()
