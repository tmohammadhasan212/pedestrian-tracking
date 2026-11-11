from tracker.cli import main

if __name__ == "__main__":
    main(["audit-data", *__import__("sys").argv[1:]])
