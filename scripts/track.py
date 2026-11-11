from tracker.cli import main

if __name__ == "__main__":
    main(["track", *__import__("sys").argv[1:]])
