# RafeekBot Docker Setup

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Setup

1. Create `.env` file in project root:
```bash
PINECONE_API_KEY=your_key
GEMINI_API_KEY=your_key
```

2. Start the application:
```bash
docker-compose up
```

3. Access at `http://localhost:8080`

## Commands

**Stop:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f rafeek-bot
```

**Rebuild image:**
```bash
docker-compose build --no-cache
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Change `8080:8080` to `8081:8080` in docker-compose.yml |
| Missing API keys | Verify `.env` file exists with required keys |
| Container won't start | Run `docker-compose logs rafeek-bot` for details |
