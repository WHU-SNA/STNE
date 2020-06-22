from stne import STNE
from utils import parameter_parser, args_printer


def main():
    args = parameter_parser()
    args_printer(args)

    stne = STNE(args)
    stne.fit()
    stne.save_emb()


if __name__ == '__main__':
    main()
