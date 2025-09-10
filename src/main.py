from utils import load_config
from animation import generate_animation


def main():
    cfg = load_config("config.json")
    generate_animation(cfg)


if __name__ == "__main__":
    main()
