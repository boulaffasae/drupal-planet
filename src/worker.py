import json as json_module
from workers import WorkerEntrypoint, Request, Response
from bs4 import BeautifulSoup
import hashlib


class Default(WorkerEntrypoint):
    async def fetch(self, request: Request) -> Response:
        if request.method != "POST":
            return Response(
                json_module.dumps({"error": "Method not allowed"}),
                status=405,
                headers={"Content-Type": "application/json"},
            )

        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            return Response(
                json_module.dumps({"error": "Invalid JSON body"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        if not url:
            return Response(
                json_module.dumps({"error": "Missing 'url' field"}),
                status=400,
                headers={"Content-Type": "application/json"},
            )

        try:
            feed_response = await fetch(url)
            content = await feed_response.text()
        except Exception as e:
            return Response(
                json_module.dumps({"error": f"Failed to fetch feed: {str(e)}"}),
                status=502,
                headers={"Content-Type": "application/json"},
            )

        soup = BeautifulSoup(content, "xml")
        items = soup.find_all("item")

        if not items:
            return Response(
                json_module.dumps({"error": "No items found in feed"}),
                status=422,
                headers={"Content-Type": "application/json"},
            )

        api_key = getattr(self.env, "MISTRAL_API_KEY")

        processed = 0
        errors = []

        for item in items:
            title_tag = item.find("title")
            link_tag = item.find("link")
            description_tag = item.find("description")

            title = title_tag.get_text(strip=True) if title_tag else ""
            link = link_tag.get_text(strip=True) if link_tag else ""
            description = description_tag.get_text(strip=True) if description_tag else ""

            if not link:
                errors.append({"item": title, "error": "missing link, skipped"})
                continue

            text = f"{title}\n{description}".strip()
            if not text:
                errors.append({"item": link, "error": "empty content, skipped"})
                continue

            # Vectorize IDs must be alphanumeric + hyphens; hash the URL to get a stable ID
            vector_id = hashlib.md5(link.encode()).hexdigest()

            try:
                response = await fetch(
                    "https://api.mistral.ai/v1/embeddings",
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    body=json_module.dumps({
                        "model": "mistral-embed",
                        "input": [text],
                    }),
                )
                data = await response.json()
                embedding = data["data"][0]["embedding"]

                await self.env.VECTOR_DB.insert([{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "title": title,
                        "link": link,
                        "description": description[:500],  # Vectorize metadata size limit
                    },
                }])
                processed += 1
            except Exception as e:
                errors.append({"item": link, "error": str(e)})
                continue

        return Response(
            json_module.dumps({
                "status": "success",
                "processed": processed,
                "total": len(items),
                "errors": errors,
            }),
            headers={"Content-Type": "application/json"},
        )
