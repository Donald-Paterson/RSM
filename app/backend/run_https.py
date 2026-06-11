import uvicorn

from app.backend.config import settings


def main() -> None:
    missing_files = [
        path
        for path in (settings.ssl_certfile, settings.ssl_keyfile)
        if not path.is_file()
    ]

    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"HTTPS certificate file(s) not found: {missing}")

    uvicorn.run(
        "app.backend.main:app",
        host=settings.https_host,
        port=settings.https_port,
        ssl_certfile=str(settings.ssl_certfile),
        ssl_keyfile=str(settings.ssl_keyfile),
    )


if __name__ == "__main__":
    main()
