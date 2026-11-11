from tracker.cli import main

if __name__ == "__main__":
    main(["evaluate", *__import__("sys").argv[1:]])
